"""Small async OpenAI client shared by VibeSwipe's AI features."""

import asyncio
import json
import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_IMAGES_URL = "https://api.openai.com/v1/images/generations"


class OpenAIResponseError(Exception):
    """Raised when OpenAI cannot produce a usable response."""


def _headers() -> dict[str, str]:
    api_key = get_settings().openai_api_key
    if not api_key:
        raise OpenAIResponseError("OPENAI_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _extract_output_text(response: dict[str, Any]) -> str:
    for output in response.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise OpenAIResponseError("OpenAI response did not contain output text")


async def generate_structured(
    *,
    model: str,
    schema_name: str,
    schema: dict[str, Any],
    instructions: str,
    input_text: str,
    max_output_tokens: int,
    reasoning_effort: str,
    retries: int = 1,
) -> dict[str, Any]:
    """Generate and decode a strict JSON-schema response through the Responses API."""
    payload = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    OPENAI_RESPONSES_URL,
                    headers=_headers(),
                    json=payload,
                )

            if response.status_code == 200:
                return json.loads(_extract_output_text(response.json()))

            error = OpenAIResponseError(
                f"OpenAI Responses API error: {response.status_code}"
            )
            if response.status_code not in (408, 409, 429) and response.status_code < 500:
                raise error
            last_error = error
        except (httpx.HTTPError, json.JSONDecodeError, OpenAIResponseError) as error:
            last_error = error
            if isinstance(error, OpenAIResponseError) and str(error) == "OPENAI_API_KEY is not configured":
                raise

        if attempt < retries:
            await asyncio.sleep(2**attempt)

    raise last_error or OpenAIResponseError("OpenAI request failed")


async def generate_image_base64(
    prompt: str,
    *,
    max_retries: int = 1,
) -> str:
    """Generate one square JPEG cover and return its base64-encoded bytes."""
    settings = get_settings()
    payload = {
        "model": settings.openai_image_model,
        "prompt": prompt,
        "size": "1024x1024",
        "quality": "medium",
        "output_format": "jpeg",
        "output_compression": 85,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    OPENAI_IMAGES_URL,
                    headers=_headers(),
                    json=payload,
                )

            if response.status_code == 200:
                image = response.json().get("data", [{}])[0].get("b64_json")
                if isinstance(image, str) and image:
                    return image
                raise OpenAIResponseError("OpenAI image response did not contain base64 image data")

            error = OpenAIResponseError(
                f"OpenAI Images API error: {response.status_code}"
            )
            if response.status_code not in (408, 409, 429) and response.status_code < 500:
                raise error
            last_error = error
        except (httpx.HTTPError, OpenAIResponseError) as error:
            last_error = error
            if isinstance(error, OpenAIResponseError) and str(error) == "OPENAI_API_KEY is not configured":
                raise

        if attempt < max_retries:
            await asyncio.sleep(2**attempt)

    raise last_error or OpenAIResponseError("OpenAI image request failed")
