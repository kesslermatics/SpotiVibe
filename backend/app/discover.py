import json
import asyncio
import logging
import httpx
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.1-pro-preview:generateContent?key={settings.gemini_api_key}"
)

SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

SYSTEM_PROMPT = """You are a music recommendation expert. The user will describe a mood, vibe, activity, or specific song preferences.

Your job is to recommend exactly 50 songs that perfectly match their request.

Respond ONLY with valid JSON in this exact format, nothing else:
{
  "mood_summary": "A short 1-sentence description of the vibe/mood you interpreted",
  "songs": [
    {"title": "Song Name", "artist": "Artist Name"},
    ...
  ]
}

Rules:
- Always recommend exactly 50 songs
- Mix well-known and lesser-known tracks
- Consider the language/culture of the request (e.g. German input → include some German/European artists)
- Only output valid JSON, no markdown, no explanation"""

SYSTEM_PROMPT_WITH_PLAYLIST = """You are a music recommendation expert. The user will describe a mood, vibe, activity, or specific song preferences.

Your job is to recommend exactly 50 songs that perfectly match their request.
You must also generate a creative, catchy playlist name and a short playlist description that captures the vibe.

Respond ONLY with valid JSON in this exact format, nothing else:
{
  "mood_summary": "A short 1-sentence description of the vibe/mood you interpreted",
  "playlist_name": "A creative, catchy playlist name (max 5 words)",
  "playlist_description": "A short fun description for the playlist (1-2 sentences)",
  "songs": [
    {"title": "Song Name", "artist": "Artist Name"},
    ...
  ]
}

Rules:
- Always recommend exactly 50 songs
- Mix well-known and lesser-known tracks
- Consider the language/culture of the request (e.g. German input → include some German/European artists)
- Only output valid JSON, no markdown, no explanation
- The playlist name should be creative and match the vibe, not generic"""


async def ask_gemini(
    prompt: str,
    context_songs: list[str] | None = None,
    on_repeat_songs: list[dict] | None = None,
    save_to_playlist: bool = False,
) -> dict:
    """Ask Gemini to interpret the mood and suggest songs."""
    logger.info(f"[Discover] ask_gemini called - prompt: '{prompt[:100]}...', context_songs: {len(context_songs) if context_songs else 0}, on_repeat: {len(on_repeat_songs) if on_repeat_songs else 0}, save_to_playlist: {save_to_playlist}")
    system = SYSTEM_PROMPT_WITH_PLAYLIST if save_to_playlist else SYSTEM_PROMPT
    parts = [{"text": system}]

    # If the user provided a playlist as context, include it
    if context_songs:
        song_list = "\n".join(f"- {s}" for s in context_songs)
        parts.append({
            "text": (
                f"The user has a playlist with these songs as reference:\n{song_list}\n\n"
                "Use this playlist as inspiration for the style/mood/genre. "
                "Recommend songs that fit the same vibe but DO NOT include any of these songs in your recommendations. "
                "Avoid duplicates completely."
            )
        })

    # If "include my taste" is enabled, add user's top tracks as style reference
    if on_repeat_songs:
        taste_list = "\n".join(f"- {s['title']} by {s['artist']}" for s in on_repeat_songs[:30])
        parts.append({
            "text": (
                f"Here are the user's current favorite/most-played songs (their taste profile):\n{taste_list}\n\n"
                "Use this to understand the user's music taste and style preferences. "
                "Your recommendations should align with their taste while still following the user's request. "
                "DO NOT include any of these songs in your recommendations."
            )
        })

    parts.append({"text": f"User request: {prompt}"})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 2.0,
            "maxOutputTokens": 8192,
        },
    }

    logger.debug(f"[Discover] Sending request to Gemini API...")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(GEMINI_URL, json=payload)

    if resp.status_code != 200:
        logger.error(f"[Discover] Gemini API error: status={resp.status_code}, body={resp.text[:500]}")
        raise Exception(f"Gemini API error: {resp.status_code} – {resp.text}")

    logger.debug(f"[Discover] Gemini response received, status={resp.status_code}")
    data = resp.json()
    
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error(f"[Discover] Unexpected Gemini response structure: {e}, data={json.dumps(data)[:500]}")
        raise Exception(f"Unexpected Gemini response: {e}")

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove first line
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
        logger.info(f"[Discover] Gemini returned {len(result.get('songs', []))} songs, mood: '{result.get('mood_summary', '')[:50]}'")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"[Discover] Failed to parse Gemini JSON: {e}, raw text: {text[:500]}")
        raise Exception(f"Invalid JSON from Gemini: {e}")


async def search_spotify(query: str, spotify_token: str) -> dict | None:
    """Search Spotify for a track and return the first result."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            SPOTIFY_SEARCH_URL,
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {spotify_token}"},
        )

    if resp.status_code != 200:
        logger.warning(f"[Discover] Spotify search failed for '{query}': status={resp.status_code}, body={resp.text[:200]}")
        return None

    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        return None

    track = items[0]
    album_images = track.get("album", {}).get("images", [])

    return {
        "title": track["name"],
        "artist": ", ".join(a["name"] for a in track["artists"]),
        "spotify_url": track["external_urls"].get("spotify"),
        "album_image": album_images[0]["url"] if album_images else None,
        "preview_url": track.get("preview_url"),
        "spotify_uri": track.get("uri"),
    }


async def discover_songs(
    prompt: str,
    spotify_token: str,
    context_songs: list[str] | None = None,
    on_repeat_songs: list[dict] | None = None,
    save_to_playlist: bool = False,
) -> dict:
    """Full pipeline: Gemini interprets mood → Spotify searches for each song (parallel)."""
    logger.info(f"[Discover] discover_songs called - prompt: '{prompt[:80]}...'")
    
    try:
        gemini_result = await ask_gemini(prompt, context_songs, on_repeat_songs, save_to_playlist)
    except Exception as e:
        logger.error(f"[Discover] Gemini call failed: {e}")
        raise

    async def fetch_song(song: dict) -> dict:
        query = f"{song['title']} {song['artist']}"
        spotify_data = await search_spotify(query, spotify_token)
        if spotify_data:
            return spotify_data
        return {
            "title": song["title"],
            "artist": song["artist"],
            "spotify_url": None,
            "album_image": None,
            "preview_url": None,
            "spotify_uri": None,
        }

    logger.info(f"[Discover] Starting Spotify search for {len(gemini_result.get('songs', []))} songs...")
    songs = await asyncio.gather(
        *(fetch_song(song) for song in gemini_result.get("songs", []))
    )
    
    found_count = sum(1 for s in songs if s.get("spotify_uri"))
    logger.info(f"[Discover] Spotify search complete: {found_count}/{len(songs)} songs found")

    # Deduplicate by Spotify URI (Gemini may suggest the same song twice with different spelling)
    seen_uris: set[str] = set()
    unique_songs = []
    for song in songs:
        uri = song.get("spotify_uri")
        if uri:
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
        unique_songs.append(song)

    logger.info(f"[Discover] Returning {len(unique_songs)} unique songs (deduplicated from {len(songs)})")
    return {
        "mood_summary": gemini_result.get("mood_summary", ""),
        "playlist_name": gemini_result.get("playlist_name"),
        "playlist_description": gemini_result.get("playlist_description"),
        "songs": unique_songs,
    }
