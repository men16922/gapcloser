# Submission checklist

Deadline: 2026-10-30 10:00 PDT (KST 10-31 02:00). Target: submit 10-29.

## Done locally (verified 2026-10-05)

- [x] App runs Nemotron 3 Super 120B on **Nebius Token Factory** (rules requirement), key in `.env` (gitignored, never committed).
- [x] `make check`: 33 tests green, also from a fresh clone with only `requirements.txt` + pytest/httpx.
- [x] `make demo` reproduces the recorded scenarios from a fresh clone (Newton, no LLM needed).
- [x] Dashboard: static (`make site` → `site/index.html`, self-contained ~4 MB) and live (`make serve`).
- [x] Docker image builds from the repo and ran a full live run on Token Factory (actuator_gain 0.8 + camera pitch 3° → estimates 0.799 / 3.0 → 100%, 1 LLM call).
- [x] Demo film: `make film` → `video/out/tether_film.mp4` (1080p, 2:00, 31 MB): motion graphics rendered in headless Chrome, ElevenLabs narration, generated music bed; every number from `runs/proof/*.json` and the recorded runs. Opens on real footage (EV-RealPhys). The earlier screen-capture cut is `make video`.
- [ ] Upload the film to YouTube or Vimeo (public or unlisted) for the Devpost video URL.
- [x] README, LICENSE (Apache-2.0), Devpost draft (`docs/submission/DEVPOST.md`), storyboard.
- [x] Token Factory spend so far ≈ $0.06 of $25.

## Needs you (outward-facing or account actions)

- [ ] Watch `video/out/tether_demo.mp4`. Optionally re-record with your own voice (the storyboard has the script) and upload to YouTube/Vimeo (public or unlisted, per Devpost rules).
- [ ] Create a **public GitHub repo** and push (`git remote add origin … && git push -u origin main`). Check that `.env` is not in `git ls-files`.
- [ ] Optional live demo: Hugging Face Space (free) or Nebius CPU VM (~$0.06/h). Steps: `docs/deploy/DEPLOY.md`. Add `NEBIUS_API_KEY` as a platform secret.
- [ ] Optional static demo: GitHub Pages from `site/`.
- [ ] Devpost: paste `docs/submission/DEVPOST.md`, add the video URL, repo URL, (live demo URL), choose the Physical AI track, list team members.
- [ ] Share the dashboard artifact only if you want it public (it is private now).

## Quick re-verification before submitting

```bash
make check
.venv/bin/python -m agent.llm --provider tokenfactory --models   # key still valid
make serve   # open http://localhost:8000, run one custom world
git ls-files | grep -c '^.env$'   # must print 0
```
