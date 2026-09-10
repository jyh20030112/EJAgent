"""Validate wheel and sdist contents produced by the packaging configuration."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def only_match(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {pattern!r} artifact in {directory}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: package_artifact_smoke.py DIST_DIRECTORY")

    dist_dir = Path(sys.argv[1])
    wheel = only_match(dist_dir, "ejagent_core-*.whl")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for required in (
            "ejagent/planning/model.py",
            "ejagent/contracts/planning.py",
            "ejagent/evaluation/workspace.py",
        ):
            if required not in names:
                raise AssertionError(f"wheel is missing {required}")
        if "ejagent/py.typed" not in names:
            raise AssertionError("wheel is missing ejagent/py.typed")
        if any(name.startswith("simagentplg/") for name in names):
            raise AssertionError("wheel unexpectedly contains the legacy package")
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()

    if "Name: ejagent-core\n" not in metadata:
        raise AssertionError("wheel metadata has the wrong distribution name")
    if "Version: 0.6.1\n" not in metadata:
        raise AssertionError("wheel metadata has the wrong distribution version")
    if "Provides-Extra: mcp" not in metadata:
        raise AssertionError("wheel metadata does not declare the mcp extra")
    if "Provides-Extra: anthropic" not in metadata:
        raise AssertionError("wheel metadata does not declare the anthropic extra")
    fastmcp_requirements = [
        line
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: fastmcp")
    ]
    if not fastmcp_requirements or not all(
        "extra == 'mcp'" in line for line in fastmcp_requirements
    ):
        raise AssertionError("fastmcp must only be required by the mcp extra")
    anthropic_requirements = [
        line
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: anthropic")
    ]
    if not anthropic_requirements or not all(
        "extra == 'anthropic'" in line for line in anthropic_requirements
    ):
        raise AssertionError("anthropic must only be required by its extra")
    for removed_package in (
        "ejagent/agent/",
        "ejagent/handlers/",
        "ejagent/middleware/",
        "ejagent/plugins/",
        "ejagent/session/",
    ):
        if any(name.startswith(removed_package) for name in names):
            raise AssertionError(f"wheel unexpectedly contains {removed_package}")
    for removed_module in (
        "ejagent/providers/base.py",
        "ejagent/providers/openai.py",
    ):
        if removed_module in names:
            raise AssertionError(f"wheel unexpectedly contains {removed_module}")


if __name__ == "__main__":
    main()
