"""Playlist-cover generation and Spotify cover upload."""

import base64
import logging
from io import BytesIO

import httpx
from PIL import Image

from app.openai_helper import generate_image_base64

logger = logging.getLogger(__name__)


async def generate_playlist_cover(
    playlist_name: str,
    mood_summary: str,
    playlist_description: str | None = None,
    max_retries: int = 2,
) -> str | None:
    """Generate a Spotify-compatible JPEG cover with OpenAI's image model."""
    description = playlist_description or mood_summary
    prompt = f"""Create a stylish square album cover for a Spotify playlist.

Playlist name: "{playlist_name}"
Mood/Vibe: {mood_summary}
Description: {description}

Requirements:
- No text, letters, or words anywhere in the image
- Match colours and style to the mood
- Modern, cinematic music-streaming aesthetic
- Use abstract art, landscapes, objects, neon lights, or cinematic scenes
- Use a visual metaphor for the theme
- Do not depict real human faces or bodies"""

    try:
        logger.info("[CoverGen] Generating cover for '%s'", playlist_name)
        image_data = await generate_image_base64(
            prompt,
            max_retries=max(0, max_retries - 1),
        )
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image = image.resize((640, 640), Image.Resampling.LANCZOS)

        jpeg_bytes = b""
        for quality in (85, 70, 55, 40):
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            jpeg_bytes = buffer.getvalue()
            if len(jpeg_bytes) <= 256 * 1024:
                break

        if len(jpeg_bytes) > 256 * 1024:
            image = image.resize((500, 500), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=50)
            jpeg_bytes = buffer.getvalue()

        logger.info("[CoverGen] Prepared %s KB JPEG", len(jpeg_bytes) // 1024)
        return base64.b64encode(jpeg_bytes).decode("utf-8")
    except Exception as error:
        logger.warning("[CoverGen] OpenAI cover generation failed: %s", error)
        return None


async def upload_playlist_cover(
    playlist_id: str,
    image_base64: str,
    spotify_token: str,
) -> bool:
    """Upload a base64-encoded JPEG to Spotify as a playlist cover."""
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/images"
    image_bytes_size = len(image_base64) * 3 // 4
    logger.info(
        "[CoverGen] Uploading cover (~%s KB) to playlist %s",
        image_bytes_size // 1024,
        playlist_id,
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(
                url,
                content=image_base64,
                headers={
                    "Authorization": f"Bearer {spotify_token}",
                    "Content-Type": "image/jpeg",
                },
            )
        if response.status_code in (200, 202):
            logger.info("[CoverGen] Successfully uploaded cover for playlist %s", playlist_id)
            return True

        logger.error("[CoverGen] Spotify upload failed: %s", response.status_code)
        return False
    except Exception as error:
        logger.error("[CoverGen] Upload failed: %s", error)
        return False
