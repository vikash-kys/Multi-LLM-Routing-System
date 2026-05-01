import time
import os
import logging
from typing import Optional

import httpx

from app.models import ModelConfig, Provider, Response

logger = logging.getLogger(__name__)


async def send_request(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
) -> Response:
    """
    Unified interface for sending requests to any LLM provider.
    Returns a standardized Response object regardless of provider.
    """
    start = time.monotonic()

    try:
        if model_config.provider == Provider.OPENAI:
            result = await _call_openai(prompt, model_config, system_prompt, max_tokens)
        elif model_config.provider == Provider.ANTHROPIC:
            result = await _call_anthropic(prompt, model_config, system_prompt, max_tokens)
        elif model_config.provider == Provider.OLLAMA:
            result = await _call_ollama(prompt, model_config, system_prompt, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {model_config.provider}")

        latency_ms = int((time.monotonic() - start) * 1000)
        result.latency_ms = latency_ms
        result.cost_usd = model_config.estimate_cost(result.input_tokens, result.output_tokens)
        return result

    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.error(f"Error calling {model_config.model_id}: {e}")
        return Response(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            model_id=model_config.model_id,
            provider=model_config.provider.value,
            error=str(e),
        )


async def _call_openai(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: Optional[str],
    max_tokens: int,
) -> Response:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_config.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]["message"]["content"]
    usage = data["usage"]

    return Response(
        text=choice,
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        latency_ms=0,
        cost_usd=0.0,
        model_id=model_config.model_id,
        provider=model_config.provider.value,
    )


async def _call_anthropic(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: Optional[str],
    max_tokens: int,
) -> Response:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    body = {
        "model": model_config.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        body["system"] = system_prompt

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

    text = data["content"][0]["text"]
    usage = data["usage"]

    return Response(
        text=text,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        latency_ms=0,
        cost_usd=0.0,
        model_id=model_config.model_id,
        provider=model_config.provider.value,
    )


async def _call_ollama(
    prompt: str,
    model_config: ModelConfig,
    system_prompt: Optional[str],
    max_tokens: int,
) -> Response:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model_config.model_id,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    text = data["message"]["content"]
    # Ollama doesn't always return token counts — estimate from characters
    input_tokens = len(prompt) // 4
    output_tokens = len(text) // 4

    return Response(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=0,
        cost_usd=0.0,
        model_id=model_config.model_id,
        provider=model_config.provider.value,
    )
