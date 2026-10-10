"""Tether runs as a NeMo Agent Toolkit workflow, and NAT's event stream sees every tool call of the agent.
Offline: replays a recorded Nemotron 3 Super run. Needs `pip install -e integrations/nat_tether` (skipped otherwise)."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("nat")

ROOT = Path(__file__).resolve().parents[3]


async def _run(config: Path, source: str):
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.loader import load_config

    from nat.builder.context import Context

    steps = []
    async with WorkflowBuilder.from_config(config=load_config(config)) as builder:
        workflow = await builder.build()
        async with workflow.run(source) as runner:
            Context.get().intermediate_step_manager.subscribe(steps.append)
            out = await runner.result(to_type=str)
    return out, steps


def test_recorded_nemotron_run_under_nat_reports_each_tool_call(monkeypatch):
    monkeypatch.chdir(ROOT)
    out, steps = asyncio.run(_run(ROOT / "integrations/nat_tether/configs/tether_replay.yml", "lab-bench"))
    res = json.loads(out)
    assert res["chosen_by"].startswith("Nemotron agent") and res["agent_steps"] >= 4
    assert {"mu_eff", "actuator_gain"} <= set(res["structure"])
    tools = [s.payload.name for s in steps if s.payload.event_type == "TOOL_END" and (s.payload.name or "").startswith("tether.")]
    assert "tether.fit_hypothesis" in tools and tools[-1] == "tether.commit"
    assert len(tools) == res["agent_steps"]
