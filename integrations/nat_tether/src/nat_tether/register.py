"""Tether as an NVIDIA NeMo Agent Toolkit (NAT) workflow.

`nat run --config_file integrations/nat_tether/configs/tether.yml --input lab-bench` calibrates a Studio sample (or
a robot log at a path) with Tether's Nemotron tool agent. NAT supplies the LLM configuration (any `llms:` entry,
here Nemotron 3 Super on Nebius Token Factory), the runner and front ends (`nat run`, `nat serve`), and the event
stream: every tool call the agent makes (decel_profile, perception_check, fit_hypothesis, probe_real, commit) is
pushed as a NAT TOOL_START/TOOL_END step, so NAT's profiler and observability exporters see the agent's path.

The agent loop and the least-squares fitter are Tether's own (agent/tool_agent.py, studio/fit.py): NAT core ships
no agent, and the numbers must come from the fitter, never from the model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.intermediate_step import IntermediateStepPayload, IntermediateStepType, StreamEventData

REPO = Path(__file__).resolve().parents[4]


class TetherCalibrateConfig(FunctionBaseConfig, name="tether_calibrate"):
    """Calibrate a simulator from a robot log: structure, fitted values with 90% intervals, the patterns left in the
    data, and the next experiment worth running."""

    llm_name: LLMRef | None = Field(default=None, description="The LLM that drives the tool agent; none = offline fit.")
    repo: str = Field(default=str(REPO), description="Path to the Tether repository (imports studio/ and agent/).")
    n_boot: int = Field(default=30, description="Bootstrap refits for the intervals.")
    record: str | None = Field(default=None, description="Append every LLM response to this JSON file (replayable).")
    replay: str | None = Field(default=None, description="Replay a recorded run from this JSON file instead of calling the LLM.")


def _llm_from(config, builder: Builder):
    """Tether's OpenAI-compatible client, built from a NAT `llms:` entry (base_url, model, key)."""
    from agent.llm import OpenAICompatLLM

    c = builder.get_llm_config(config.llm_name)
    key = c.api_key.get_secret_value() if getattr(c, "api_key", None) else os.environ.get("NEBIUS_API_KEY", "")
    if not key:
        raise RuntimeError("No API key: set NEBIUS_API_KEY or api_key in the NAT llms entry.")
    model = c.model_name
    return OpenAICompatLLM(c.base_url, key, {"diagnose": model, "summarize": model, "vision": model}, f"nat:{config.llm_name}",
                           temperature=c.temperature if getattr(c, "temperature", None) is not None else 0.2)


def _load(source: str):
    from studio.session import load

    samples = Path(REPO) / "studio" / "samples"
    path = samples / f"{source}.csv" if (samples / f"{source}.csv").exists() else Path(source).expanduser()
    if not path.exists():
        names = sorted(p.stem for p in samples.glob("*.csv"))
        raise ValueError(f"'{source}' is neither a sample ({', '.join(names)}) nor a file")
    s = load(path.read_text(errors="replace"), path.name, path.stem)
    domain = {"brake-log": "driving", "press-line": "factory"}.get(path.stem, "robot")
    s.meta["domain"] = domain
    return s


def summarize(res) -> dict:
    from studio import structure

    cal = res.calibration
    return {
        "chosen_by": cal.chosen_by,
        "structure": cal.structure,
        "model": {k: v for k, v in cal.model.items() if v is not None},
        "intervals_90": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in cal.intervals.items()},
        "stop_residual_rms_m": cal.residuals["stop_residual_rms_m"],
        "left_unexplained": structure.describe(cal.residuals.get("patterns") or {"findings": [], "note": "not checked"}),
        "predicted_success": {"current_sim": cal.predicted["before"]["median"], "calibrated": cal.predicted["after"]["median"]},
        "next_experiment": [{"stop_near_m": s["predicted_stop_m"], "why": s["why"]} for s in res.next_experiment["suggestions"]],
        "agent_steps": res.agent["steps"] if res.agent else 0,
    }


@register_function(config_type=TetherCalibrateConfig)
async def tether_calibrate(config: TetherCalibrateConfig, builder: Builder):
    if config.repo not in sys.path:
        sys.path.insert(0, config.repo)
    from agent.llm import RecordedLLM, RecordingLLM

    if config.replay:
        llm = RecordedLLM.from_file(Path(config.replay))
    else:
        llm = _llm_from(config, builder) if config.llm_name else None
        if llm is not None and config.record:
            llm = RecordingLLM(llm, Path(config.record))

    async def _calibrate(source: str) -> str:
        """Calibrate from `source`: a Studio sample name (lab-bench, short-reach, brake-log, press-line) or the path
        of a robot log (CSV/JSON: command, stop, optional target, perceived, track). Returns JSON."""
        from studio.pipeline import analyze

        steps = Context.get().intermediate_step_manager

        def on_event(e: dict) -> None:  # each agent tool call -> a NAT tool step (start and end)
            if e.get("type") != "agent_step":
                return
            st = e["step"]
            name = f"tether.{st.get('tool', 'step')}"
            start = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START, name=name,
                                            data=StreamEventData(input=st.get("args")))
            steps.push_intermediate_step(start)
            steps.push_intermediate_step(IntermediateStepPayload(
                UUID=start.UUID, event_type=IntermediateStepType.TOOL_END, name=name,
                span_event_timestamp=start.event_timestamp, data=StreamEventData(input=st.get("args"), output=st.get("result"))))

        session = _load(source.strip())
        res = analyze(session, llm, on_event=on_event, n_boot=config.n_boot)
        return json.dumps(summarize(res), indent=1, default=str)

    yield FunctionInfo.from_fn(_calibrate, description=_calibrate.__doc__)
