# Submission checklist

Deadline: 2026-10-30 10:00 PDT (KST 10-31 02:00). Target: submit 10-29.

## Done locally (verified 2026-10-05)

- [x] App runs Nemotron 3 Super 120B on **Nebius Token Factory** (rules requirement), key in `.env` (gitignored, never committed).
- [x] `make check`: 33 tests green, also from a fresh clone with only `requirements.txt` + pytest/httpx.
- [x] `make demo` reproduces the recorded scenarios from a fresh clone (Newton, no LLM needed).
- [x] Live demo on Google Cloud Run: https://tether-454741001655.us-central1.run.app (`make deploy`, verified 2026-10-11: Nemotron analysis, Ask, video tracking, retraining).
- [x] Docker image builds from the repo and ran a full live run on Token Factory (actuator_gain 0.8 + camera pitch 3° → estimates 0.799 / 3.0 → 100%, 1 LLM call).
- [x] Demo film: `make film` → `video/out/tether_film.mp4` (1080p, 2:28): motion graphics rendered in headless Chrome around ~68 s of the app in action, ElevenLabs narration, generated music bed; every number from `runs/proof/*.json` and the recorded runs. Opens on real footage (EV-RealPhys).
- [x] README, LICENSE (Apache-2.0), Devpost draft (`docs/submission/DEVPOST.md`).
- [x] Token Factory spend so far ≈ $0.06 of $25.

## Needs you (outward-facing or account actions)

- [ ] Upload: everything is in `submission/` (`make submission`); step by step in `docs/submission/SUBMIT.md` (YouTube public, then Devpost).
- [x] Public GitHub repo: https://github.com/men16922/tether (`.env` is not in `git ls-files`).
- [ ] Devpost: paste `docs/submission/DEVPOST.md`, add the video URL, the repo URL and the live demo URL, choose the Physical AI track, list team members.

## Quick re-verification before submitting

```bash
make check
.venv/bin/python -m agent.llm --provider tokenfactory --models   # key still valid
make serve   # open http://localhost:8000, run one custom world
git ls-files | grep -c '^.env$'   # must print 0
```
