"""Tether: tie your simulator to the real world.

The application, one package per job:

    agent/   the reasoning agent: Nemotron on Nebius Token Factory (tool calling), the LLM client, Cosmos Reason eyes
    sim/     physics: the push task, NVIDIA Newton worlds, MuJoCo hidden worlds, parameters and replays
    studio/  the calibration engine behind Studio: video tracking, fit and intervals, retraining, Newton check, export
    server/  FastAPI: serves the three pages and the API (Server-Sent Events for live runs)
    web/     the three pages (Overview, Studio, Benchmark), their shared design system, and the page builder
    eval/    benchmarks and `make prove`: every headline number, recomputed offline

Repository paths live in `tether.paths`.
"""
