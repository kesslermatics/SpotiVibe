"""
Vibe Roast – AI-generated sarcastic music profile.

Flow:
1. Fetch user's top 50 tracks + top artists (long_term)
2. Bulk-fetch audio features for all tracks
3. Compute average audio features
4. Extract top genres from top artists
5. Send everything to OpenAI for a sarcastic roast
"""

import asyncio
import json
import logging
import re

import httpx

from app.config import get_settings
from app.openai_helper import generate_structured

settings = get_settings()
logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"

ROAST_SCHEMA = {
    "type": "object",
    "properties": {
        "persona": {"type": "string", "minLength": 1},
        "roast": {"type": "string", "minLength": 1},
    },
    "required": ["persona", "roast"],
    "additionalProperties": False,
}


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


async def ask_openai_roast(
    top_tracks: list[str],
    top_artists: list[str],
    top_genres: list[str],
    avg_features: dict,
) -> dict:
    """Use GPT-6 Luna to roast the user's music taste."""
    features_text = "\n".join(f"- {key}: {value}" for key, value in avg_features.items())
    tracks_text = "\n".join(f"- {track}" for track in top_tracks[:20])
    artists_text = "\n".join(f"- {artist}" for artist in top_artists[:15])
    genres_text = ", ".join(top_genres[:10])
    prompt = f"""Here is a Spotify user's data:

TOP SONGS:
{tracks_text}

TOP ARTISTS:
{artists_text}

TOP GENRES: {genres_text}

AUDIO FEATURES (averages, 0.0 to 1.0 except tempo):
{features_text}

Create a short, punchy roasty persona title (maximum five words) and a brutal but funny, non-offensive roast in exactly three sentences. Reference specific artists, genres, or features."""
    return await generate_structured(
        model=settings.openai_utility_model,
        schema_name="vibe_roast",
        schema=ROAST_SCHEMA,
        instructions="You are a sarcastic, witty music critic. Be funny rather than mean.",
        input_text=prompt,
        max_output_tokens=1024,
        reasoning_effort="none",
        retries=2,
    )


def _try_repair_json(text: str) -> dict | None:
    """Attempt to repair truncated JSON from OpenAI."""
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

    # 4. Extract data for OpenAI
    top_track_names = [
        f"{t['name']} - {', '.join(a['name'] for a in t['artists'])}"
        for t in top_tracks_raw
    ]
    top_artist_names = [a["name"] for a in top_artists_raw]
    top_genres = extract_top_genres(top_artists_raw)

    # 5. Ask OpenAI for the roast
    logger.info("Vibe Roast: Asking OpenAI for roast...")
    roast_result = await ask_openai_roast(
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
