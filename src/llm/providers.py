"""Language-model back ends behind one small interface.

Everything runs locally through Ollama (https://ollama.com); no data leaves the machine and
no API key or paid service is needed.

* :class:`OllamaProvider`: a local open model served by Ollama (default ``llama3.1:8b``; any
  pulled chat model, e.g. ``qwen3:14b``, can be chosen).
* :class:`OllamaEmbedder`: local text embeddings from Ollama (default ``nomic-embed-text``),
  used for dense retrieval in the RAG assistant.

Settings come from environment variables, optionally from a git-ignored ``.env`` file in
the project root (see ``.env.example``). Nothing here is needed by the ML pipeline: when
no provider is reachable, the assistant still retrieves and shows the sources.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np

from src.config import PROJECT_ROOT

DEFAULTS = {
    "LLM_PROVIDER": "ollama",
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_MODEL": "llama3.1:8b",
    "OLLAMA_EMBED_MODEL": "nomic-embed-text",
}


def _dotenv() -> dict[str, str]:
    p = PROJECT_ROOT / ".env"
    out: dict[str, str] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def setting(name: str) -> str:
    """Environment variable, else ``.env``, else the default."""
    return os.environ.get(name) or _dotenv().get(name) or DEFAULTS.get(name, "")


class LLMError(RuntimeError):
    """The model could not produce an answer (not configured, unreachable, refused)."""


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict = field(default_factory=dict)


class Provider(Protocol):
    name: str
    model: str

    def available(self) -> tuple[bool, str]: ...

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> Completion: ...


# --------------------------------------------------------------------------- Ollama
class OllamaProvider:
    """A local model served by Ollama (``ollama serve``; ``ollama pull llama3.1:8b``)."""

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, temperature: float = 0.1) -> None:
        self.model = model or setting("OLLAMA_MODEL")
        self.host = (host or setting("OLLAMA_HOST")).rstrip("/")
        self.temperature = temperature

    def models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=3.0)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return []

    def chat_models(self) -> list[str]:
        """Pulled models that can write answers (embedding-only models left out)."""
        return [m for m in self.models() if "embed" not in m.lower()]

    def available(self) -> tuple[bool, str]:
        names = self.models()
        if not names:
            return False, f"Ollama is not running at {self.host} (start the Ollama app or `ollama serve`)"
        if self.model not in names and f"{self.model}:latest" not in names:
            return False, f"Ollama model {self.model!r} is not pulled (`ollama pull {self.model}`)"
        return True, f"Ollama · {self.model} (local)"

    def complete(self, system: str, prompt: str, max_tokens: int = 1500) -> Completion:
        ok, why = self.available()
        if not ok:
            raise LLMError(why)
        start = time.perf_counter()
        try:
            r = httpx.post(f"{self.host}/api/chat", timeout=300.0, json={
                "model": self.model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "options": {"temperature": self.temperature, "num_predict": max_tokens, "num_ctx": 8192},
            })
            r.raise_for_status()
            body = r.json()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        return Completion(body["message"]["content"].strip(), self.name, self.model, time.perf_counter() - start,
                          int(body.get("prompt_eval_count", 0)), int(body.get("eval_count", 0)))


class OllamaEmbedder:
    """Sentence embeddings from Ollama, L2-normalised (cosine similarity = dot product)."""

    def __init__(self, model: str | None = None, host: str | None = None, batch: int = 32) -> None:
        self.model = model or setting("OLLAMA_EMBED_MODEL")
        self.host = (host or setting("OLLAMA_HOST")).rstrip("/")
        self.batch = batch

    def available(self) -> tuple[bool, str]:
        names = OllamaProvider(host=self.host).models()
        if not names:
            return False, f"Ollama is not running at {self.host}"
        if self.model not in names and f"{self.model}:latest" not in names:
            return False, f"embedding model {self.model!r} is not pulled (`ollama pull {self.model}`)"
        return True, f"{self.model} (Ollama)"

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        # nomic-embed-text expects task prefixes; harmless for other models
        prefix = {"document": "search_document: ", "query": "search_query: "}.get(kind, "") \
            if "nomic" in self.model else ""
        out = []
        for i in range(0, len(texts), self.batch):
            r = httpx.post(f"{self.host}/api/embed", timeout=300.0,
                           json={"model": self.model, "input": [prefix + t for t in texts[i:i + self.batch]]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        v = np.asarray(out, dtype=np.float32)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def get_provider(name: str | None = None, model: str | None = None) -> Provider | None:
    """``ollama`` (a local model; ``model`` overrides ``OLLAMA_MODEL``) or ``none`` (default: ``LLM_PROVIDER``)."""
    name = (name or setting("LLM_PROVIDER")).lower()
    if name == "ollama":
        return OllamaProvider(model=model)
    return None
