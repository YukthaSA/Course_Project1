"""
LLM provider adapters - one function per backend, all with the same signature:

    call_<provider>(cfg, model, system, prompt) -> str

Each one makes a single blocking HTTP request and raises on any failure (HTTP error, timeout,
empty body). Routing, fallback and tracing live in BaseAgent.call_llm; nothing here knows
about pipeline stages.

    Gemini     -> generativelanguage.googleapis.com  (cloud)
    Groq       -> api.groq.com/openai/v1             (cloud, very fast Llama / GPT-OSS inference)
    OpenAI     -> api.openai.com/v1                  (cloud)
    Anthropic  -> api.anthropic.com/v1               (cloud)
    Ollama     -> localhost:11434/api/chat           (local, offline, any installed model)
"""
import json
import urllib.error
import urllib.request
from typing import Any, Dict

TEMPERATURE = 0.2


def _post(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "CourseProject1/1.0", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")[:200].replace("\n", " ")
        raise RuntimeError(f"HTTP {err.code} {detail}") from None


def _chat_messages(system: str, prompt: str):
    return [*([{"role": "system", "content": system}] if system else []), {"role": "user", "content": prompt}]


def _answer(data: Dict[str, Any]) -> str:
    """Text of an OpenAI-style chat completion; a reply cut off by the token limit is an error, not an answer."""
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("reply truncated by the token limit (finish_reason=length)")
    return choice["message"]["content"] or ""


def call_groq(cfg, model: str, system: str, prompt: str) -> str:
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        {"model": model, "messages": _chat_messages(system, prompt), "temperature": TEMPERATURE,
         "max_completion_tokens": 8192},  # reasoning models spend part of the cap on thinking
        {"Authorization": f"Bearer {cfg.GROQ_API_KEY}"},
        timeout=60,
    )
    return _answer(data)


def call_openai(cfg, model: str, system: str, prompt: str) -> str:
    data = _post(
        "https://api.openai.com/v1/chat/completions",
        {"model": model, "messages": _chat_messages(system, prompt), "temperature": TEMPERATURE,
         "max_tokens": 8192},
        {"Authorization": f"Bearer {cfg.OPENAI_API_KEY}"},
        timeout=90,
    )
    return _answer(data)


def call_gemini(cfg, model: str, system: str, prompt: str) -> str:
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": TEMPERATURE},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload,
        {"x-goog-api-key": cfg.GOOGLE_API_KEY},
        timeout=90,
    )
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def call_anthropic(cfg, model: str, system: str, prompt: str) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    data = _post(
        "https://api.anthropic.com/v1/messages",
        payload,
        {"x-api-key": cfg.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
        timeout=90,
    )
    return "".join(b.get("text", "") for b in data.get("content", []))


def call_ollama(cfg, model: str, system: str, prompt: str) -> str:
    """Native /api/chat (unlike the OpenAI-compatible route it honours num_ctx / num_predict)."""
    data = _post(
        f"{cfg.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        {
            "model": model,
            "messages": _chat_messages(system, prompt),
            "stream": False,
            "keep_alive": "10m",  # keep the model loaded between pipeline stages
            "options": {"temperature": TEMPERATURE, "num_ctx": cfg.OLLAMA_NUM_CTX, "num_predict": 2048},
        },
        {},
        timeout=cfg.OLLAMA_TIMEOUT,
    )
    return data["message"]["content"]


PROVIDERS = {
    "gemini": call_gemini,
    "google": call_gemini,
    "groq": call_groq,
    "openai": call_openai,
    "anthropic": call_anthropic,
    "ollama": call_ollama,
}
