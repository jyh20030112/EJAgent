from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

from ejagent.contracts import (
    CancellationSource,
    CancellationToken,
    EvaluationCriterion,
    EvaluationPlan,
    RunCancelledError,
)
from ejagent.evaluation import (
    CommandEvidenceSource,
    EvidenceUnavailable,
    GoalEvaluator,
    WorkspaceEvidenceSource,
    command_succeeded,
)
from ejagent.kernel.trajectory import (
    CheckpointSignal,
    CheckpointTrigger,
    TrajectoryCost,
)


def checkpoint() -> CheckpointSignal:
    return CheckpointSignal(
        "run",
        CheckpointTrigger.BASELINE,
        0,
        TrajectoryCost(),
        evaluation_plan=EvaluationPlan(
            "Tests pass",
            "v1",
            (EvaluationCriterion("tests", "Tests pass", "command", ("tests",)),),
        ),
    )


class TestWorkspaceEvidence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "feature.py"
        self.path.write_text("value = 1\n")
        self.workspace = WorkspaceEvidenceSource(self.root, ("feature.py",))
        self.token = CancellationToken()

    async def test_command_cache_invalidated_by_code_and_host_revision(self) -> None:
        command = CommandEvidenceSource(
            self.workspace,
            (
                sys.executable,
                "-B",
                "-c",
                "from pathlib import Path; p=Path('count'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); exec(Path('feature.py').read_text()); print(value); raise SystemExit(value != 1)",
            ),
        )
        evaluator = GoalEvaluator(
            sources={"tests": command}, verifiers={"command": command_succeeded}
        )
        before = await evaluator.evaluate("cp0", checkpoint(), cancellation=self.token)
        self.assertEqual(before.requirements[0].status.value, "pass")
        await evaluator.evaluate("cp1", checkpoint(), cancellation=self.token)
        self.assertEqual((self.root / "count").read_text(), "x")
        self.path.write_text("value = 2\n")
        after = await evaluator.evaluate("cp2", checkpoint(), cancellation=self.token)
        self.assertEqual(after.requirements[0].status.value, "fail")
        self.assertNotEqual(
            before.evidence["tests"].revision, after.evidence["tests"].revision
        )
        self.assertTrue(after.invalidated_refs)
        command.invalidate()
        await evaluator.evaluate("cp3", checkpoint(), cancellation=self.token)
        self.assertEqual((self.root / "count").read_text(), "xxx")
        evaluator.close_run("run")
        await evaluator.evaluate("cp4", checkpoint(), cancellation=self.token)
        self.assertEqual((self.root / "count").read_text(), "xxxx")
        evaluator.close_run("run")

    async def test_command_cannot_verify_a_workspace_it_changed(self) -> None:
        command = CommandEvidenceSource(
            self.workspace,
            (
                sys.executable,
                "-c",
                "from pathlib import Path; Path('feature.py').write_text('changed')",
            ),
        )
        with self.assertRaisesRegex(EvidenceUnavailable, "workspace changed"):
            await command.read(checkpoint(), cancellation=self.token)

    async def test_output_is_bounded_and_timeout_is_unavailable(self) -> None:
        command = CommandEvidenceSource(
            self.workspace,
            (sys.executable, "-c", "print('x'*100000)"),
            max_output_bytes=128,
        )
        snapshot = await command.read(checkpoint(), cancellation=self.token)
        self.assertEqual(len(snapshot.value["stdout"]), 128)
        self.assertTrue(snapshot.value["output_truncated"])
        slow = CommandEvidenceSource(
            self.workspace,
            (sys.executable, "-c", "import time; time.sleep(10)"),
            timeout_seconds=0.05,
        )
        with self.assertRaisesRegex(EvidenceUnavailable, "timed out"):
            await slow.read(checkpoint(), cancellation=self.token)

    async def test_cancellation_reaps_command(self) -> None:
        command = CommandEvidenceSource(
            self.workspace,
            (
                sys.executable,
                "-c",
                "import os,time; from pathlib import Path; Path('pid').write_text(str(os.getpid())); time.sleep(30)",
            ),
        )
        source = CancellationSource()
        running = asyncio.create_task(
            command.read(checkpoint(), cancellation=source.token)
        )
        async with asyncio.timeout(2):
            while not (self.root / "pid").exists():  # noqa: ASYNC110 -- wait for an external process, not an asyncio producer
                await asyncio.sleep(0.01)
        pid = int((self.root / "pid").read_text())
        source.cancel("stop")
        with self.assertRaises(RunCancelledError):
            await running
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_absent_file_and_symlink_escape(self) -> None:
        workspace = WorkspaceEvidenceSource(self.root, ("missing.py",))
        snapshot = await workspace.read(checkpoint(), cancellation=self.token)
        self.assertEqual(
            snapshot.value["files"]["missing.py"], {"exists": False, "text": None}
        )
        (self.root / "escape").symlink_to(self.root.parent)
        escaped = WorkspaceEvidenceSource(self.root, ("escape/file",))
        with self.assertRaises(EvidenceUnavailable):
            await escaped.read(checkpoint(), cancellation=self.token)
        with self.assertRaises(ValueError):
            WorkspaceEvidenceSource(self.root, ("../outside",))

    async def test_workspace_total_bound(self) -> None:
        (self.root / "second.py").write_text("x" * 12)
        workspace = WorkspaceEvidenceSource(
            self.root, ("feature.py", "second.py"), max_bytes=15
        )
        with self.assertRaisesRegex(EvidenceUnavailable, "total size"):
            await workspace.read(checkpoint(), cancellation=self.token)
