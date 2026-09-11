from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.planned_feature import DemoFeatureActor, DemoFeaturePlanner, run_feature


class TestPlannedFeature(unittest.IsolatedAsyncioTestCase):
    async def test_real_code_failure_drives_replan_and_completion_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = await run_feature(
                root, actor=DemoFeatureActor(), planner_model=DemoFeaturePlanner()
            )
            self.assertTrue(result.result.succeeded)
            rejected = [
                r for r in result.audit_records if r.kind == "completion_rejected"
            ]
            self.assertEqual(len(rejected), 1)
            updates = [
                r for r in result.audit_records if r.kind == "execution_plan_updated"
            ]
            self.assertEqual([r.payload["version"] for r in updates], [2, 3, 4])
            self.assertIn("Boundary-case failure", updates[1].payload["reason"])
            # Check behavior of the actual generated file, not merely its source text.
            namespace = {}
            exec((root / "intervals.py").read_text(), namespace)
            self.assertEqual(namespace["overlap_seconds"](0, 1, 2, 8), 0)
            self.assertEqual(namespace["overlap_seconds"](0, 5, 2, 8), 3)
            self.assertFalse(namespace["overlapped"](0, 2, 2, 8))
            contexts = [
                json.loads(line)
                for line in (root / "actor-contexts.jsonl").read_text().splitlines()
            ]
            for request in contexts:
                projected = [
                    json.loads(m["content"])["trajectory_context"]
                    for m in request["messages"]
                    if m.get("source", "").startswith("trajectory:")
                ]
                self.assertEqual(len(projected), 1)
                state = projected[0]
                task = next(
                    json.loads(m["content"])["task_context"]
                    for m in request["messages"]
                    if m.get("source") == "planning"
                )
                self.assertEqual(state["state_status"], "available")
                self.assertEqual(state["checkpoint"], task["latest_checkpoint"])
                self.assertEqual(state["for_turn"], request["request"])
                self.assertNotIn("cycle_suspected", json.dumps(state))
            unchanged = json.loads(
                next(
                    m["content"]
                    for m in contexts[1]["messages"]
                    if m.get("source") == "trajectory:state"
                )
            )["trajectory_context"]
            self.assertFalse(unchanged["requirements"]["regression"])
            self.assertIsNone(unchanged["feedback"])
            final_state = json.loads(
                next(
                    m["content"]
                    for m in contexts[-1]["messages"]
                    if m.get("source") == "trajectory:state"
                )
            )["trajectory_context"]
            self.assertTrue(final_state["requirements"]["regression"])
            self.assertEqual(
                final_state["progress"]["current_requirement_coverage"], 1.0
            )
            self.assertIsNone(final_state["feedback"])
            recovery = contexts[3]
            self.assertTrue(
                any(m.get("source") == "completion_audit" for m in recovery["messages"])
            )
            self.assertIn("AssertionError", json.dumps(recovery))
            self.assertTrue((root / "contexts.md").exists())
            summary = json.loads((root / "result.json").read_text())
            self.assertEqual(summary["planner_requests"], 1)
            self.assertEqual(summary["final_plan"]["version"], 4)
