# Tether — Sim2Real self-closing agent (Nebius x NVIDIA hackathon)
# Gate must stay OFFLINE + DETERMINISTIC: no GPU, no network, no Nebius/Token Factory calls.
.PHONY: check test lint smoke-local demo dashboard replays franka-mesh serve docker site video studio-samples studio-record studio-bench

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

PY_SRC := $(shell find agent sim eval dashboard server video studio -name '*.py' 2>/dev/null)

check: lint test

lint:
	@for f in docs/AGENT_BRIEF.md docs/STATUS.md docs/NEXT_PLAN.md; do test -f $$f || { echo "missing $$f"; exit 1; }; done
	@if [ -n "$(PY_SRC)" ]; then $(PY) -m py_compile $(PY_SRC) && echo "py_compile OK"; else echo "no python sources yet"; fi

test:
	@if [ -d tests ] && ls tests/test_*.py >/dev/null 2>&1; then $(PY) -m pytest -q tests; else echo "no tests yet"; fi

smoke-local: check

# record Newton demo runs (local CPU, no network) and build the self-contained dashboard
demo:
	$(PY) -m eval.record_demo
	$(PY) -m dashboard.build

dashboard:
	$(PY) -m dashboard.build

# 3D viewer data for the recorded bundle (Newton on CPU, no LLM, no network), then rebuild the dashboard
replays:
	$(PY) -m eval.record_demo --replays-only
	$(PY) -m dashboard.build

# regenerate dashboard/assets/franka_fr3.json from the Newton Franka asset (decimated visual meshes)
franka-mesh:
	$(PY) -m dashboard.franka_mesh

# Studio sample data: Newton robot logs + Newton-rendered phone videos with ground truth (studio/samples)
studio-samples:
	$(PY) -m studio.samples
	$(PY) -m studio.video_sample

# record the samples through the Studio API (Nemotron agent when GAPCLOSER_LLM / --llm is set), rebuild pages
studio-record:
	$(PY) -m studio.record --llm $${GAPCLOSER_LLM:-tokenfactory}
	$(PY) -m dashboard.build

# active vs passive data collection: real pushes needed to reach 95% (analytic, 50 worlds, ~30 s)
studio-bench:
	$(PY) -m eval.studio_bench --worlds 50 --json runs/bench/studio_bench_analytic.json

# live server on http://localhost:8000 (Studio at /studio) (GAPCLOSER_LLM=local|tokenfactory|none)
serve: dashboard
	$(PY) -m uvicorn server.app:app --host 0.0.0.0 --port $${PORT:-8000}

docker:
	docker build -t gapcloser .

# static, self-contained page for GitHub Pages (recorded runs only)
site: dashboard
	mkdir -p site && cp dashboard/dist/gapcloser.standalone.html site/index.html

# demo video (Chrome + ffmpeg + macOS say); start `make serve` first to include the live scene
video: dashboard
	$(PY) -m video.make_video

# ===== overnight harness targets (append to your Makefile) =====
# The overnight runner + helpers are the Single Source of Truth in the overnight-harness
# PLUGIN; this repo does NOT vendor them. These targets resolve the installed plugin at
# runtime and invoke its runner against THIS repo. Per-repo STATE stays here:
#   scripts/overnight/overnight-settings.json  — Claude permission boundary
#   scripts/overnight/opencode.json            — opencode permission boundary
#   .codex/rules/overnight.rules               — Codex command rules
#   scripts/overnight/PROMPT.md                — optional per-repo prompt override (else plugin default)
#   scripts/overnight/{logs,STOP,DONE}         — runtime state
#   docs/LESSONS.md                            — the actor's memory surface (committed)
#
# The loop's commit gate is $GATE_CMD (default `make check`). Define a `check` target that proves
# correctness OFFLINE + DETERMINISTICALLY and allow-list it in scripts/overnight/overnight-settings.json.
#
# Select the engine with ENGINE=claude|codex|opencode|agy|kiro. Default stays Claude.
ENGINE ?= claude

# Model routing (Claude engine; defaults refreshed in 1.7.0). Actor and critic both run on
# Opus 5.5. Its effort default is medium (one level below Opus 5), so both are pinned to high.
# Blank = the CLI's own default. These are per-repo policy, so edit them here rather than in
# the plugin.
CLAUDE_MODEL ?= claude-opus-5-5
CLAUDE_EFFORT ?= high
CLAUDE_CRITIC_MODEL ?= claude-opus-5-5
OVERNIGHT_CRITIC_MODEL ?=
CLAUDE_CRITIC_EFFORT ?= high
CLAUDE_FALLBACK_MODEL ?=
export CLAUDE_MODEL CLAUDE_EFFORT CLAUDE_CRITIC_MODEL OVERNIGHT_CRITIC_MODEL CLAUDE_CRITIC_EFFORT CLAUDE_FALLBACK_MODEL

# Codex routing is opt-in; preserve installed model/effort settings when blank.
# Astra preset: make overnight-codex-once CODEX_MODEL=gpt-6-astra
CODEX_MODEL ?=
CODEX_EFFORT ?=
CODEX_CRITIC_EFFORT ?=
export CODEX_MODEL CODEX_EFFORT CODEX_CRITIC_EFFORT

# HARNESS_ROOT resolution (env override → per-repo pin → highest installed version). This mirrors
# the plugin's bin/harness-locate.sh; override ad hoc with `make overnight HARNESS_ROOT=/path`.
HARNESS_ROOT ?= $(shell \
  if [ -n "$$OVERNIGHT_HARNESS_ROOT" ] && [ -d "$$OVERNIGHT_HARNESS_ROOT/templates/scripts/overnight" ]; then \
    echo "$$OVERNIGHT_HARNESS_ROOT"; \
  elif [ -n "$$OVERNIGHT_HARNESS_ROOT" ] && [ -d "$$OVERNIGHT_HARNESS_ROOT/plugins/overnight-harness/templates/scripts/overnight" ]; then \
    echo "$$OVERNIGHT_HARNESS_ROOT/plugins/overnight-harness"; \
  elif [ -f .claude/harness-config.json ] && grep -q '"harness_root"' .claude/harness-config.json; then \
    pin="$$(sed -n 's/.*"harness_root"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' .claude/harness-config.json | head -1)"; \
    if [ -d "$$pin/templates/scripts/overnight" ]; then echo "$$pin"; \
    elif [ -d "$$pin/plugins/overnight-harness/templates/scripts/overnight" ]; then echo "$$pin/plugins/overnight-harness"; fi; \
  else \
    { \
      ls -d $$HOME/.claude/plugins/cache/overnight-harness/overnight-harness/*/ 2>/dev/null; \
      find $$HOME/.codex/plugins/cache -path '*/overnight-harness/*' -type d 2>/dev/null; \
      [ -d $$HOME/.gemini/antigravity-cli/plugins/overnight-harness ] && echo $$HOME/.gemini/antigravity-cli/plugins/overnight-harness; \
      [ -d $$HOME/.cache/opencode/node_modules/opencode-overnight-harness ] && echo $$HOME/.cache/opencode/node_modules/opencode-overnight-harness; \
    } | while read d; do [ -d "$$d/templates/scripts/overnight" ] && echo "$$d"; done | sort -V | tail -1; \
  fi)

# OVN_SRC = runner + helpers (in the plugin); OVN = per-repo state (in this repo).
# NB: no inline comments on these := lines — make would fold the gap into the value.
OVN_SRC := $(HARNESS_ROOT:%/=%)/templates/scripts/overnight
OVN := scripts/overnight

_harness-guard:
	@test -x "$(OVN_SRC)/run.sh" || { \
	  echo "overnight-harness not found (resolved HARNESS_ROOT='$(HARNESS_ROOT)')."; \
	  echo "Install the plugin, or pass HARNESS_ROOT=/path/to/plugin, or re-run /harness-init."; \
	  exit 1; }

overnight: _harness-guard           ## run the unattended loop (caffeinate keeps macOS awake)
	OVERNIGHT_ENGINE=$(ENGINE) caffeinate -dimsu $(OVN_SRC)/run.sh &
overnight-watch: overnight          ## start the loop and tail its log
	@sleep 1; tail -f $(OVN)/logs/runner.log
overnight-once: _harness-guard      ## single iteration (smoke test the loop)
	OVERNIGHT_ENGINE=$(ENGINE) $(OVN_SRC)/run.sh --once
overnight-claude-once: _harness-guard
	OVERNIGHT_ENGINE=claude $(OVN_SRC)/run.sh --once
overnight-codex-once: _harness-guard
	OVERNIGHT_ENGINE=codex $(OVN_SRC)/run.sh --once
overnight-opencode-once: _harness-guard
	OVERNIGHT_ENGINE=opencode $(OVN_SRC)/run.sh --once
overnight-agy-once: _harness-guard
	OVERNIGHT_ENGINE=agy $(OVN_SRC)/run.sh --once
overnight-kiro-once: _harness-guard
	OVERNIGHT_ENGINE=kiro $(OVN_SRC)/run.sh --once
overnight-stop:                     ## graceful stop after the current iteration
	@touch $(OVN)/STOP && echo "STOP created — loop will exit after current iteration"
overnight-clean:                    ## clear STOP/DONE sentinels before the next run
	@rm -f $(OVN)/STOP $(OVN)/DONE && echo "cleared STOP/DONE"
overnight-status: _harness-guard    ## aggregate iteration status across lanes
	@bash $(OVN_SRC)/status.sh
overnight-logs:                     ## tail the runner log
	@mkdir -p $(OVN)/logs; touch $(OVN)/logs/runner.log; tail -f $(OVN)/logs/runner.log
overnight-dashboard: _harness-guard ## tmux dashboard (falls back to status.sh)
	@bash $(OVN_SRC)/dashboard.sh
overnight-ledger-check: _harness-guard ## validate event history before trusting status/report
	@python3 $(OVN_SRC)/lib/ledger.py check $(OVN)/logs/events.jsonl
overnight-ledger-state: _harness-guard ## project deterministic per-mission state as JSON
	@python3 $(OVN_SRC)/lib/ledger.py project $(OVN)/logs/events.jsonl
overnight-trajectory: _harness-guard ## render causal node/edge trajectory; optional MISSION and FORMAT=json
	@python3 $(OVN_SRC)/lib/trajectory.py $(OVN)/logs/events.jsonl \
	  $(if $(MISSION),--mission "$(MISSION)",) --format "$(or $(FORMAT),text)"
overnight-resume: _harness-guard       ## resume MISSION with DECISION=approve|reject
	@test -x "$(OVN_SRC)/resume.sh" || { echo "installed overnight-harness does not provide resume.sh"; exit 1; }
	@test -n "$(MISSION)" || { echo "MISSION=<mission-id> is required"; exit 2; }
	@test "$(DECISION)" = "approve" -o "$(DECISION)" = "reject" || { echo "DECISION=approve|reject is required"; exit 2; }
	@HARNESS_REPO_ROOT="$(CURDIR)" bash $(OVN_SRC)/resume.sh "$(MISSION)" --"$(DECISION)"
overnight-provenance-compare: _harness-guard ## compare two manifests; optional LEFT_RESULT/RIGHT_RESULT
	@test -n "$(LEFT)" -a -n "$(RIGHT)" || { echo "LEFT=<manifest> and RIGHT=<manifest> are required"; exit 2; }
	@python3 $(OVN_SRC)/lib/provenance.py compare --left "$(LEFT)" --right "$(RIGHT)" \
	  --left-result "$(LEFT_RESULT)" --right-result "$(RIGHT_RESULT)"
overnight-where:                    ## print the resolved plugin location (debug)
	@echo "HARNESS_ROOT = $(HARNESS_ROOT)"; echo "runner       = $(OVN_SRC)/run.sh"

.PHONY: overnight overnight-watch overnight-once overnight-claude-once overnight-codex-once overnight-opencode-once overnight-agy-once overnight-kiro-once overnight-stop overnight-clean overnight-status overnight-logs overnight-dashboard overnight-ledger-check overnight-ledger-state overnight-trajectory overnight-resume overnight-provenance-compare overnight-where _harness-guard
# ===== end overnight harness targets =====
