"""
Playlist Cover Generation using Gemini Image Generation
"""
import base64
import logging
import httpx
from io import BytesIO

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Gemini Image Generation endpoint
GEMINI_IMAGE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.1-flash-image-preview:generateContent?key={settings.gemini_api_key}"
)


async def generate_playlist_cover(
    playlist_name: str,
    mood_summary: str,
    playlist_description: str | None = None,
) -> str | None:
    """
    Generate a playlist cover image using Gemini.
    Returns base64-encoded JPEG string ready for Spotify API, or None on failure.
    
    Spotify requirements:
    - Base64-encoded JPEG
    - Square image recommended
    - Max 256KB
    """
    # Build a creative prompt for the image
    desc = playlist_description or mood_summary
    prompt = f"""Create a stylish album cover for a Spotify playlist.

Playlist name: "{playlist_name}"
Mood/Vibe: {mood_summary}
Description: {desc}

Requirements:
- NO text, NO letters, NO words anywhere in the image
- Colors and style should match the mood perfectly
- Modern, cinematic aesthetic suitable for a music streaming app
- Square format, visually striking
- Can include people, objects, scenes that represent the vibe
- Be creative and thematic: if it's about success/CEO vibes, show luxury items, a man in a suit, cigars, fancy desk. If it's about sadness, show rain, lonely scenes. Match the THEME.

Create an image that looks like a real album cover and captures the ESSENCE of this playlist."""

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }

    try:
        logger.info(f"[CoverGen] Generating cover for '{playlist_name}'...")
        
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(GEMINI_IMAGE_URL, json=payload)

        if resp.status_code != 200:
            logger.error(f"[CoverGen] Gemini API error: {resp.status_code} - {resp.text[:300]}")
            return None

        data = resp.json()
        
        # Extract the image from the response
        # Gemini returns images in candidates[0].content.parts[] with inlineData
        candidates = data.get("candidates", [])
        if not candidates:
            logger.warning("[CoverGen] No candidates in Gemini response")
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        
        for part in parts:
            if "inlineData" in part:
                inline_data = part["inlineData"]
                mime_type = inline_data.get("mimeType", "")
                image_data = inline_data.get("data", "")
                
                if image_data:
                    logger.info(f"[CoverGen] Got image, mimeType={mime_type}, size={len(image_data)} chars")
                    
                    # Always process through PIL to ensure correct size for Spotify (max 256KB)
                    try:
                        from PIL import Image
                        
                        raw_bytes = base64.b64decode(image_data)
                        img = Image.open(BytesIO(raw_bytes))
                        
                        # Convert to RGB if necessary (e.g., PNG with alpha)
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        
                        # Resize to 640x640 (Spotify recommended)
                        img = img.resize((640, 640), Image.Resampling.LANCZOS)
                        
                        # Save as JPEG, progressively reduce quality until under 256KB
                        jpeg_bytes = None
                        for quality in [85, 70, 55, 40]:
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG", quality=quality)
                            jpeg_bytes = buffer.getvalue()
                            if len(jpeg_bytes) <= 256 * 1024:
                                logger.info(f"[CoverGen] Compressed to {len(jpeg_bytes)} bytes at quality={quality}")
                                break
                        
                        if jpeg_bytes and len(jpeg_bytes) > 256 * 1024:
                            # Still too big - resize smaller
                            img = img.resize((500, 500), Image.Resampling.LANCZOS)
                            buffer = BytesIO()
                            img.save(buffer, format="JPEG", quality=50)
                            jpeg_bytes = buffer.getvalue()
                            logger.info(f"[CoverGen] Resized to 500x500, final size={len(jpeg_bytes)} bytes")
                        
                        jpeg_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                        return jpeg_b64
                        
                    except ImportError:
                        logger.warning("[CoverGen] PIL not installed, returning raw image data")
                        return image_data
                    except Exception as e:
                        logger.error(f"[CoverGen] Image conversion failed: {e}")
                        return image_data

        logger.warning("[CoverGen] No image found in Gemini response parts")
        return None

    except Exception as e:
        logger.error(f"[CoverGen] Failed to generate cover: {e}")
        return None


async def upload_playlist_cover(
    playlist_id: str,
    image_base64: str,
    spotify_token: str,
) -> bool:
    """
    Upload a cover image to a Spotify playlist.
    
    Args:
        playlist_id: Spotify playlist ID
        image_base64: Base64-encoded JPEG image (no data: prefix)
        spotify_token: Valid Spotify access token
    
    Returns:
        True on success, False on failure
    """
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/images"
    
    # Check size before upload
    image_bytes_size = len(image_base64) * 3 // 4  # Approximate decoded size
    logger.info(f"[CoverGen] Uploading cover (~{image_bytes_size // 1024}KB) to playlist {playlist_id}")
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.put(
                url,
                content=image_base64,
                headers={
                    "Authorization": f"Bearer {spotify_token}",
                    "Content-Type": "image/jpeg",
                },
            )
        
        if resp.status_code in (200, 202):
            logger.info(f"[CoverGen] Successfully uploaded cover for playlist {playlist_id}")
            return True
        else:
            logger.error(f"[CoverGen] Spotify upload failed: {resp.status_code} - {resp.text[:200]}")
            return False
            
    except Exception as e:
        import traceback
        logger.error(f"[CoverGen] Upload failed: {e}\n{traceback.format_exc()}")
        return False
