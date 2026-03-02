import json
import asyncio
import httpx
from app.config import get_settings

settings = get_settings()

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3-flash-preview:generateContent?key={settings.gemini_api_key}"
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

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(GEMINI_URL, json=payload)

    if resp.status_code != 200:
        raise Exception(f"Gemini API error: {resp.status_code} – {resp.text}")

    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove first line
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return json.loads(text)


async def search_spotify(query: str, spotify_token: str) -> dict | None:
    """Search Spotify for a track and return the first result."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            SPOTIFY_SEARCH_URL,
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {spotify_token}"},
        )

    if resp.status_code != 200:
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
    gemini_result = await ask_gemini(prompt, context_songs, on_repeat_songs, save_to_playlist)

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

    songs = await asyncio.gather(
        *(fetch_song(song) for song in gemini_result.get("songs", []))
    )

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

    return {
        "mood_summary": gemini_result.get("mood_summary", ""),
        "playlist_name": gemini_result.get("playlist_name"),
        "playlist_description": gemini_result.get("playlist_description"),
        "songs": unique_songs,
    }
