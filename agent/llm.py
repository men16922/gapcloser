"""LLM access for the agent: OpenAI-compatible providers + offline replay.

Providers (GAPCLOSER_LLM): "local" = Ollama on this Mac (NVIDIA Nemotron 3 Nano, free, for
experiments); "tokenfactory" = Nebius Token Factory (required for the submission). Model ids are
resolved at runtime from a hint (e.g. "nemotron super") by listing the provider's models. RecordedLLM replays saved responses so tests and the
offline gate never touch the network or spend credits. RecordingLLM wraps a live LLM and saves
every exchange as a fixture for later replay.

Live check:  .venv/bin/python -m agent.llm --provider local --models
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from the repo's .env into os.environ (existing variables win)."""
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


load_dotenv()

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/")

# role -> model hint per provider (all words must appear in the model id, case-insensitive)
PROVIDER_HINTS = {
    "tokenfactory": {"diagnose": "nemotron super", "summarize": "nemotron nano", "vision": "nemotron omni"},
    "local": {"diagnose": "nemotron nano 30b", "summarize": "nemotron nano 4b", "vision": "nemotron nano 30b"},
}
DEFAULT_HINTS = {
    "diagnose": os.environ.get("GAPCLOSER_DIAG_MODEL", PROVIDER_HINTS["tokenfactory"]["diagnose"]),
    "summarize": os.environ.get("GAPCLOSER_NANO_MODEL", PROVIDER_HINTS["tokenfactory"]["summarize"]),
    "vision": os.environ.get("GAPCLOSER_VISION_MODEL", PROVIDER_HINTS["tokenfactory"]["vision"]),
}


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    def complete(self, role: str, messages: list[dict], schema: dict | None = None) -> LLMResponse: ...


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.by_model[r.model] = self.by_model.get(r.model, 0) + 1


def pick_model(ids: list[str], hint: str) -> str | None:
    """First id containing every word of the hint (case-insensitive); exact id wins."""
    if hint in ids:
        return hint
    words = hint.lower().split()
    matches = sorted(i for i in ids if all(w in i.lower() for w in words))
    return matches[0] if matches else None


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, hints: dict[str, str], provider: str, temperature: float = 0.2,
                 timeout: float = 180.0, max_tokens: int = 2048):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.provider = provider
        self.hints = hints
        self.temperature = temperature
        self.max_tokens = max_tokens  # hard cap: a runaway generation must not stall the loop
        self.usage = Usage()
        self._ids: list[str] | None = None

    @classmethod
    def tokenfactory(cls, api_key: str | None = None, hints: dict[str, str] | None = None) -> OpenAICompatLLM:
        key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY is not set. See docs/setup/NEBIUS_SETUP.md.")
        return cls(BASE_URL, key, {**DEFAULT_HINTS, **(hints or {})}, "tokenfactory")

    @classmethod
    def local(cls, hints: dict[str, str] | None = None) -> OpenAICompatLLM:
        return cls(OLLAMA_URL, "ollama", {**PROVIDER_HINTS["local"], **(hints or {})}, "local")

    def models(self) -> list[str]:
        if self._ids is None:
            self._ids = sorted(m.id for m in self.client.models.list().data)
        return self._ids

    def model_for(self, role: str) -> str:
        hint = self.hints.get(role, role)
        mid = pick_model(self.models(), hint)
        if mid is None:
            raise RuntimeError(f"no {self.provider} model matches '{hint}' for role '{role}'")
        return mid

    def complete(self, role: str, messages: list[dict], schema: dict | None = None) -> LLMResponse:
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}}
        model = self.model_for(role)
        r = self.client.chat.completions.create(model=model, messages=messages, temperature=self.temperature,
                                                max_tokens=self.max_tokens, **kwargs)
        u = r.usage
        resp = LLMResponse(r.choices[0].message.content or "", model,
                           getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
        self.usage.add(resp)
        return resp


TokenFactoryLLM = OpenAICompatLLM.tokenfactory


def make_llm(provider: str | None = None) -> OpenAICompatLLM:
    provider = provider or os.environ.get("GAPCLOSER_LLM", "local")
    if provider == "local":
        return OpenAICompatLLM.local()
    if provider == "tokenfactory":
        return OpenAICompatLLM.tokenfactory()
    raise ValueError(f"unknown provider '{provider}' (local | tokenfactory)")


class RecordedLLM:
    """Replays responses in order (per role). Raises when it runs out, so tests notice."""

    def __init__(self, responses: dict[str, list[dict]]):
        self._queues = {k: list(v) for k, v in responses.items()}
        self.usage = Usage()
        self.calls: list[tuple[str, list[dict]]] = []

    @classmethod
    def from_file(cls, path: Path) -> RecordedLLM:
        return cls(json.loads(path.read_text()))

    def complete(self, role: str, messages: list[dict], schema: dict | None = None) -> LLMResponse:
        self.calls.append((role, messages))
        q = self._queues.get(role)
        if not q:
            raise RuntimeError(f"RecordedLLM: no recorded response left for role '{role}'")
        resp = LLMResponse(**q.pop(0))
        self.usage.add(resp)
        return resp


class RecordingLLM:
    """Wraps a live LLM and appends every response to a fixture file usable by RecordedLLM."""

    def __init__(self, inner: LLM, path: Path):
        self.inner, self.path = inner, path
        self.data: dict[str, list[dict]] = json.loads(path.read_text()) if path.exists() else {}
        self.usage = getattr(inner, "usage", Usage())

    def complete(self, role: str, messages: list[dict], schema: dict | None = None) -> LLMResponse:
        r = self.inner.complete(role, messages, schema)
        self.data.setdefault(role, []).append(asdict(r))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))
        return r


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="LLM provider checks (--models spends no tokens)")
    ap.add_argument("--models", action="store_true", help="list model ids and the ids chosen per role")
    ap.add_argument("--ping", action="store_true", help="one tiny completion on the diagnose model")
    ap.add_argument("--provider", choices=["local", "tokenfactory"], default=None)
    a = ap.parse_args()
    llm = make_llm(a.provider)
    if a.models or not a.ping:
        for mid in llm.models():
            print(mid)
        print("\nroles:")
        for role in llm.hints:
            try:
                print(f"  {role:10s} -> {llm.model_for(role)}")
            except RuntimeError as e:
                print(f"  {role:10s} -> {e}")
    if a.ping:
        r = llm.complete("diagnose", [{"role": "user", "content": "Reply with the single word: ready"}])
        print(r.model, repr(r.text[:80]), r.prompt_tokens, r.completion_tokens)


if __name__ == "__main__":
    main()
