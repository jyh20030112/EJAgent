"""Explicit workspace observations and host-configured verification commands."""

from __future__ import annotations

import asyncio
import math
import os
import signal as signals
from collections.abc import Mapping
from pathlib import Path

from ejagent.contracts.control import CancellationToken
from ejagent.contracts.json import JsonObject
from ejagent.evaluation.sources import FileEvidenceSource
from ejagent.evaluation.types import EvidenceSnapshot, EvidenceUnavailable, fingerprint
from ejagent.kernel.trajectory import CheckpointSignal


class WorkspaceEvidenceSource:
    """Observe explicit relative UTF-8 paths; never discover or traverse a repo.

    Include every dependency whose change must invalidate command evidence. Missing
    configured paths are evidence of absence. Symlinks outside root are rejected.
    """

    def __init__(
        self, root: str | Path, paths: tuple[str, ...], *, max_bytes: int = 262_144
    ) -> None:
        self.root = Path(root).resolve()
        if not paths or len(paths) > 128 or len(set(paths)) != len(paths):
            raise ValueError("workspace requires 1-128 unique paths")
        for path in paths:
            if Path(path).is_absolute() or ".." in Path(path).parts or not path:
                raise ValueError(
                    "workspace paths must be relative and stay within root"
                )
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be positive")
        self.paths = tuple(paths)
        self.max_bytes = max_bytes

    async def _snapshot(
        self, signal: CheckpointSignal, cancellation: CancellationToken
    ) -> EvidenceSnapshot:
        files: dict[str, JsonObject] = {}
        revisions: dict[str, str] = {}
        size = 0
        for relative in self.paths:
            cancellation.raise_if_cancelled()
            path = self.root / relative
            if not path.resolve().is_relative_to(self.root):
                raise EvidenceUnavailable("workspace path resolves outside root")
            source = FileEvidenceSource(path, max_bytes=self.max_bytes)
            snapshot = await source.read(signal, cancellation=cancellation)
            # FileEvidenceSource exposes exactly an existence flag and UTF-8 text.
            assert isinstance(snapshot.value, Mapping)
            files[relative] = snapshot.value
            size += len(str(snapshot.value.get("text", "")).encode())
            if size > self.max_bytes:
                raise EvidenceUnavailable("workspace exceeds total size bound")
            revisions[relative] = snapshot.revision
        return EvidenceSnapshot(
            fingerprint(revisions), {"files": files}, str(self.root)
        )

    async def revision(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> str:
        return (await self._snapshot(signal, cancellation)).revision

    async def read(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> EvidenceSnapshot:
        before = await self._snapshot(signal, cancellation)
        after = await self._snapshot(signal, cancellation)
        if before.revision != after.revision:
            raise EvidenceUnavailable("workspace changed during capture")
        return after

    def close_run(self, run_id: str) -> None:
        pass


class CommandEvidenceSource:
    """Run a fixed host command, binding its output to a workspace revision.

    A Run-local revision cache avoids rerunning it at unchanged checkpoints.
    Commands are host configuration, never model-authored shell strings. Commands
    should be observational; a changed workspace during execution invalidates output.
    """

    def __init__(
        self,
        workspace: WorkspaceEvidenceSource,
        command: tuple[str, ...],
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 65_536,
    ) -> None:
        if not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("command requires non-empty argument strings")
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("command timeout must be finite and positive")
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes <= 0
        ):
            raise ValueError("max_output_bytes must be positive")
        self.workspace = workspace
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self._generation = 0
        self._cache: dict[str, EvidenceSnapshot] = {}

    def invalidate(self) -> None:
        """Host invalidation for dependencies not represented by workspace files."""
        self._generation += 1
        self._cache.clear()

    async def revision(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> str:
        return fingerprint(
            {
                "workspace": await self.workspace.revision(
                    signal, cancellation=cancellation
                ),
                "command": self.command,
                "generation": self._generation,
            }
        )

    async def read(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> EvidenceSnapshot:
        before = await self.revision(signal, cancellation=cancellation)
        cached = self._cache.get(signal.run_id)
        if cached is not None and cached.revision == before:
            return cached
        try:
            async with asyncio.timeout(self.timeout_seconds):
                result = await cancellation.run(self._run())
        except TimeoutError as exc:
            raise EvidenceUnavailable("verification command timed out") from exc
        after = await self.revision(signal, cancellation=cancellation)
        if before != after:
            raise EvidenceUnavailable(
                "workspace changed while verification command ran"
            )
        snapshot = EvidenceSnapshot(
            after,
            {**result, "dependency_revision": after},
            "command:" + self.command[0],
        )
        self._cache[signal.run_id] = snapshot
        return snapshot

    async def _run(self) -> JsonObject:
        process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=self.workspace.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        assert process.stdout is not None and process.stderr is not None

        async def drain(stream: asyncio.StreamReader) -> tuple[str, bool]:
            content = bytearray()
            truncated = False
            while chunk := await stream.read(8192):
                remaining = self.max_output_bytes - len(content)
                content.extend(chunk[:remaining])
                truncated |= len(chunk) > remaining
            return content.decode("utf-8", errors="replace"), truncated

        try:
            stdout, stderr = await asyncio.gather(
                drain(process.stdout), drain(process.stderr)
            )
            code = await process.wait()
            return {
                "exit_code": code,
                "stdout": stdout[0],
                "stderr": stderr[0],
                "output_truncated": stdout[1] or stderr[1],
            }
        finally:
            # Kill the group even if the leader exited but descendants kept pipes open.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signals.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.returncode is None:
                process.kill()
            await process.wait()

    def close_run(self, run_id: str) -> None:
        self._cache.pop(run_id, None)
