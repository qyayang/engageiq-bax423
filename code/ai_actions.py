"""
Provider-agnostic AI engagement action generator.
BAX-423 · Optional LLM enhancement layer on top of rule-based suggestions.

Control via environment variables:
    LLM_PROVIDER   = openai | anthropic | gemini | ollama   (default: unset → rule-based)
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OLLAMA_MODEL

Fallback chain: LLM → rule-based suggestions (never crashes the app).
"""
import json
import os
import requests

_TIMEOUT = 10  # seconds per API call


def _secret(key: str, default: str = "") -> str:
    """Read from os.getenv first, then st.secrets (Streamlit Cloud)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(key, default))
    except Exception:
        return default

_PROMPT_TEMPLATE = """You are a developer engagement advisor.
Given this opportunity, generate exactly 3 specific, actionable engagement suggestions for the user.

Opportunity:
- Title: {title}
- Source: {source}
- Domain: {domain}
- Description: {description}
- Stars: {stars}, Good-first-issues: {gfi}, Comments: {comments}
- User role: {persona}

Rules:
- Be specific to THIS opportunity, not generic advice.
- Do not invent facts not in the fields above.
- Each action must be completable in one session.
- Return ONLY valid JSON: {{"actions": ["action1", "action2", "action3"]}}
"""


def _build_prompt(opp: dict, persona: str) -> str:
    return _PROMPT_TEMPLATE.format(
        title=opp.get("title", "")[:120],
        source=opp.get("source", ""),
        domain=opp.get("domain", ""),
        description=(opp.get("description", "") or "")[:200],
        stars=opp.get("stars", 0),
        gfi=opp.get("good_first_issues", 0),
        comments=opp.get("comments", 0),
        persona=persona or "developer",
    )


def _parse_actions(text: str, fallback: list[str]) -> list[str]:
    """Extract actions list from LLM JSON response, with fallback."""
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        data  = json.loads(text[start:end])
        actions = data.get("actions", [])
        if isinstance(actions, list) and len(actions) >= 1:
            return [str(a) for a in actions[:3]]
    except Exception:
        pass
    return fallback


# ── Provider adapters ─────────────────────────────────────────────────────────

def _openai_actions(prompt: str, fallback: list[str]) -> list[str]:
    key = _secret("OPENAI_API_KEY")
    if not key:
        return fallback
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": _secret("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.4,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _parse_actions(text, fallback)


def _anthropic_actions(prompt: str, fallback: list[str]) -> list[str]:
    key = _secret("ANTHROPIC_API_KEY")
    if not key:
        return fallback
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": _secret("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"]
    return _parse_actions(text, fallback)


def _gemini_actions(prompt: str, fallback: list[str]) -> list[str]:
    key = _secret("GEMINI_API_KEY")
    if not key:
        return fallback
    model = _secret("GEMINI_MODEL", "gemini-1.5-flash")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_actions(text, fallback)


def _deepseek_actions(prompt: str, fallback: list[str]) -> list[str]:
    key = _secret("DEEPSEEK_API_KEY")
    if not key:
        return fallback
    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": _secret("DEEPSEEK_MODEL", "deepseek-chat"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.4,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _parse_actions(text, fallback)


def _ollama_actions(prompt: str, fallback: list[str]) -> list[str]:
    model = _secret("OLLAMA_MODEL", "llama3.1")
    host  = _secret("OLLAMA_HOST", "http://localhost:11434")
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["message"]["content"]
    return _parse_actions(text, fallback)


# ── Public API ────────────────────────────────────────────────────────────────

_PROVIDERS = {
    "openai":    _openai_actions,
    "anthropic": _anthropic_actions,
    "gemini":    _gemini_actions,
    "deepseek":  _deepseek_actions,
    "ollama":    _ollama_actions,
}


def generate_ai_actions(opp: dict, persona: str, fallback_actions: list[str]) -> list[str]:
    """
    Returns LLM-generated actions if a provider is configured, else rule-based fallback.
    Never raises — any error silently returns the fallback.
    """
    provider = _secret("LLM_PROVIDER", "").lower().strip()
    if not provider or provider not in _PROVIDERS:
        return fallback_actions

    try:
        prompt = _build_prompt(opp, persona)
        return _PROVIDERS[provider](prompt, fallback_actions)
    except Exception:
        return fallback_actions
