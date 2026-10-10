# Tether on NVIDIA NeMo Agent Toolkit

Tether's calibration agent as a [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) (NAT) workflow.
NAT provides the LLM configuration (`llms:`), the runner and front ends (`nat run`, `nat serve`) and the event
stream. Every tool call the agent makes (`decel_profile`, `perception_check`, `fit_hypothesis`, `probe_real`,
`commit`) is pushed to that stream as a TOOL_START/TOOL_END step, so NAT's profiler and observability exporters
see the agent's path. The agent loop and the least-squares fitter are Tether's own: NAT core ships no agent, and the
numbers always come from the fitter, never from the model.

```bash
pip install -r requirements.txt -e integrations/nat_tether     # nvidia-nat 1.9, Python 3.11+
export NEBIUS_API_KEY=...                                      # Nemotron 3 Super on Nebius Token Factory

nat run --config_file integrations/nat_tether/configs/tether.yml --input lab-bench          # Nemotron agent
nat run --config_file integrations/nat_tether/configs/tether_offline.yml --input press-line # no LLM
nat run --config_file integrations/nat_tether/configs/tether_replay.yml --input lab-bench   # recorded run, offline
nat serve --config_file integrations/nat_tether/configs/tether.yml                          # HTTP endpoint
```

The input is a Studio sample (`lab-bench`, `short-reach`, `brake-log`, `press-line`) or the path of a robot log
(CSV/JSON: `command`, `stop`, optional `target`, `perceived`, `track`). The output is JSON: the structure the agent
chose, fitted values with 90% bootstrap intervals, what the tracks still leave unexplained, predicted success before
and after, and the next experiment.

`recorded/lab-bench.json` is a live Nemotron 3 Super run captured with `configs/tether_record.yml` (2026-10-10, 6
tool calls). `tests/test_nat_tether.py` replays it under NAT and checks that the event stream carries each call:

```bash
python -m pytest integrations/nat_tether/tests
```

NAT's own bundled tools plugin warns that `langchain_core` is missing; Tether does not use it.
