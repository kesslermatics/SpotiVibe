"""
Vibe Roast – AI-generated sarcastic music profile.

Flow:
1. Fetch user's top 50 tracks + top artists (long_term)
2. Bulk-fetch audio features for all tracks
3. Compute average audio features
4. Extract top genres from top artists
5. Send everything to Gemini for a sarcastic roast
"""

import asyncio
import json
import logging
import re

import httpx

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.1-pro-preview:generateContent?key={settings.gemini_api_key}"
)


async def fetch_top_tracks(spotify_token: str, limit: int = 50) -> list[dict]:
    """Fetch user's top tracks (long_term for accurate profile)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SPOTIFY_API}/me/top/tracks",
            params={"limit": limit, "time_range": "long_term"},
            headers={"Authorization": f"Bearer {spotify_token}"},
        )
    if resp.status_code != 200:
        logger.warning(f"Roast: Failed to fetch top tracks: {resp.status_code}")
        return []
    return resp.json().get("items", [])


async def fetch_top_artists(spotify_token: str, limit: int = 50) -> list[dict]:
    """Fetch user's top artists (long_term)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SPOTIFY_API}/me/top/artists",
            params={"limit": limit, "time_range": "long_term"},
            headers={"Authorization": f"Bearer {spotify_token}"},
        )
    if resp.status_code != 200:
        logger.warning(f"Roast: Failed to fetch top artists: {resp.status_code}")
        return []
    return resp.json().get("items", [])


async def fetch_audio_features_bulk(
    track_ids: list[str], spotify_token: str
) -> list[dict]:
    """Bulk-fetch audio features (up to 100 IDs per request).
    
    Falls back gracefully if endpoint returns 403 (dev-mode restriction).
    """
    all_features: list[dict] = []
    headers = {"Authorization": f"Bearer {spotify_token}"}

    # Process in chunks of 100
    for i in range(0, len(track_ids), 100):
        chunk = track_ids[i : i + 100]
        ids_str = ",".join(chunk)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{SPOTIFY_API}/audio-features",
                params={"ids": ids_str},
                headers=headers,
            )
        if resp.status_code == 200:
            features = resp.json().get("audio_features", [])
            all_features.extend([f for f in features if f is not None])
        elif resp.status_code == 403:
            logger.warning("Roast: Audio features endpoint restricted (403). Using defaults.")
            return []  # Will trigger default values in compute_avg_features
        else:
            logger.warning(f"Roast: Audio features fetch failed: {resp.status_code}")

    return all_features


def compute_avg_features(features: list[dict]) -> dict:
    """Compute average audio feature values."""
    if not features:
        return {
            "danceability": 0.5,
            "energy": 0.5,
            "valence": 0.5,
            "acousticness": 0.5,
            "instrumentalness": 0.0,
            "speechiness": 0.1,
            "tempo": 120.0,
        }

    keys = [
        "danceability", "energy", "valence",
        "acousticness", "instrumentalness", "speechiness", "tempo",
    ]
    averages = {}
    for key in keys:
        values = [f[key] for f in features if key in f]
        averages[key] = round(sum(values) / len(values), 3) if values else 0.0

    return averages


def extract_top_genres(artists: list[dict], limit: int = 10) -> list[str]:
    """Extract most common genres from top artists."""
    genre_count: dict[str, int] = {}
    for artist in artists:
        for genre in artist.get("genres", []):
            genre_count[genre] = genre_count.get(genre, 0) + 1

    sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
    return [g[0] for g in sorted_genres[:limit]]


async def ask_gemini_roast(
    top_tracks: list[str],
    top_artists: list[str],
    top_genres: list[str],
    avg_features: dict,
) -> dict:
    """Ask Gemini to roast the user's music taste."""

    features_text = "\n".join(
        f"- {k}: {v}" for k, v in avg_features.items()
    )
    tracks_text = "\n".join(f"- {t}" for t in top_tracks[:20])
    artists_text = "\n".join(f"- {a}" for a in top_artists[:15])
    genres_text = ", ".join(top_genres[:10])

    prompt = f"""You are a sarcastic, witty music critic.

Here is a Spotify user's data:

TOP SONGS:
{tracks_text}

TOP ARTISTS:
{artists_text}

TOP GENRES: {genres_text}

AUDIO FEATURES (averages, 0.0 to 1.0 except tempo):
{features_text}

Your task:
1. Create a short, roasty persona title (e.g. "Sad-Girl-Indie Protagonist", "Gym-Bro Metal Enjoyer", "Mainstream NPC with Spotify-Wrapped Trauma")
2. Write a brutal but funny roast about the user's music taste in exactly 3 sentences. Be creative, sarcastic and specific!

Respond ONLY with valid JSON:
{{
  "persona": "Your creative persona title",
  "roast": "Your 3-sentence roast here."
}}

Rules:
- ONLY valid JSON, no markdown, no explanation
- The persona title should be short and punchy (max 5 words)
- The roast should be exactly 3 sentences long
- Be brutally honest but funny, not offensive
- Reference specific artists, genres or features"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.5,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }

    last_error = None
    for attempt in range(3):
        if attempt > 0:
            logger.info(f"Roast Gemini retry {attempt + 1}/3")
            await asyncio.sleep(1)

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(GEMINI_URL, json=payload)

        if resp.status_code != 200:
            last_error = f"Gemini API error: {resp.status_code} – {resp.text[:300]}"
            continue

        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            last_error = f"Unexpected Gemini response: {e}"
            continue

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            result = json.loads(text)
            if "persona" in result and "roast" in result:
                return result
            last_error = f"JSON missing required keys: {list(result.keys())}"
            continue
        except json.JSONDecodeError:
            # Try to repair truncated JSON
            repaired = _try_repair_json(text)
            if repaired and "persona" in repaired and "roast" in repaired:
                logger.info("Roast: Repaired truncated Gemini JSON")
                return repaired
            last_error = f"Invalid JSON after repair attempt: {text[:200]}"
            continue

    raise Exception(f"Gemini failed after 3 attempts: {last_error}")


def _try_repair_json(text: str) -> dict | None:
    """Attempt to repair truncated JSON from Gemini."""
    try:
        # Try extracting persona and roast via regex
        persona_m = re.search(r'"persona"\s*:\s*"([^"]+)"', text)
        roast_m = re.search(r'"roast"\s*:\s*"((?:[^"\\]|\\.)*)"?', text, re.DOTALL)
        if persona_m and roast_m:
            return {
                "persona": persona_m.group(1),
                "roast": roast_m.group(1).replace('\\n', ' ').strip(),
            }
    except Exception:
        pass
    return None


async def generate_vibe_roast(spotify_token: str) -> dict:
    """Full Vibe Roast pipeline."""
    logger.info("Vibe Roast: Starting...")

    # 1. Fetch top tracks + top artists in parallel
    top_tracks_raw, top_artists_raw = await asyncio.gather(
        fetch_top_tracks(spotify_token),
        fetch_top_artists(spotify_token),
    )

    if len(top_tracks_raw) < 5:
        raise Exception(
            "You need at least 5 top songs for a Vibe Roast. "
            "Listen to more music and try again later!"
        )

    logger.info(
        f"Vibe Roast: Got {len(top_tracks_raw)} tracks, {len(top_artists_raw)} artists"
    )

    # 2. Bulk-fetch audio features
    track_ids = [t["id"] for t in top_tracks_raw]
    audio_features = await fetch_audio_features_bulk(track_ids, spotify_token)
    logger.info(f"Vibe Roast: Got audio features for {len(audio_features)} tracks")

    # 3. Compute averages
    avg_features = compute_avg_features(audio_features)

    # 4. Extract data for Gemini
    top_track_names = [
        f"{t['name']} - {', '.join(a['name'] for a in t['artists'])}"
        for t in top_tracks_raw
    ]
    top_artist_names = [a["name"] for a in top_artists_raw]
    top_genres = extract_top_genres(top_artists_raw)

    # 5. Ask Gemini for the roast
    logger.info("Vibe Roast: Asking Gemini for roast...")
    roast_result = await ask_gemini_roast(
        top_track_names, top_artist_names, top_genres, avg_features
    )
    logger.info(f"Vibe Roast: Got persona '{roast_result.get('persona', '?')}'")

    return {
        "persona": roast_result.get("persona", "Mystery Listener"),
        "roast": roast_result.get("roast", "Could not generate a roast."),
        "audio_features": avg_features,
        "top_genres": top_genres,
        "top_artists": top_artist_names[:10],
        "track_count": len(top_tracks_raw),
    }
