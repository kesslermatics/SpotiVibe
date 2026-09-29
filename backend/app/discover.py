import asyncio
import logging

import httpx

from app.config import get_settings
from app.openai_helper import generate_structured

logger = logging.getLogger(__name__)
settings = get_settings()

SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

SONG_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "artist": {"type": "string", "minLength": 1},
    },
    "required": ["title", "artist"],
    "additionalProperties": False,
}

DISCOVER_SCHEMA = {
    "type": "object",
    "properties": {
        "mood_summary": {"type": "string"},
        "playlist_name": {"type": ["string", "null"]},
        "playlist_description": {"type": ["string", "null"]},
        "songs": {
            "type": "array",
            "items": SONG_SCHEMA,
            "minItems": 50,
            "maxItems": 50,
        },
    },
    "required": ["mood_summary", "playlist_name", "playlist_description", "songs"],
    "additionalProperties": False,
}

WRONG_INDICES_SCHEMA = {
    "type": "object",
    "properties": {
        "wrong_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        }
    },
    "required": ["wrong_indices"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a music recommendation expert. The user will describe a mood, vibe, activity, or specific song preferences.

Your job is to recommend exactly 50 songs that perfectly match their request.

Rules:
- Always recommend exactly 50 songs
- Mix well-known and lesser-known tracks
- Consider the language/culture of the request (e.g. German input → include some German/European artists)
- Do not recommend songs provided as reference context
- Do not recommend duplicate songs"""

SYSTEM_PROMPT_WITH_PLAYLIST = """You are a music recommendation expert. The user will describe a mood, vibe, activity, or specific song preferences.

Your job is to recommend exactly 50 songs that perfectly match their request. You must also generate a creative, catchy playlist name and a short playlist description that captures the vibe.

Rules:
- Always recommend exactly 50 songs
- Mix well-known and lesser-known tracks
- Consider the language/culture of the request (e.g. German input → include some German/European artists)
- Do not recommend songs provided as reference context
- Do not recommend duplicate songs
- The playlist name should be creative and match the vibe, not generic
- Playlist name: maximum five words; playlist description: one or two short sentences"""


async def ask_openai(
    prompt: str,
    context_songs: list[str] | None = None,
    on_repeat_songs: list[dict] | None = None,
    save_to_playlist: bool = False,
) -> dict:
    """Interpret a mood and suggest songs with GPT-6.1 Sol."""
    logger.info(
        "[Discover] Generating recommendations: context=%s, on_repeat=%s, save_to_playlist=%s",
        len(context_songs) if context_songs else 0,
        len(on_repeat_songs) if on_repeat_songs else 0,
        save_to_playlist,
    )
    context_blocks: list[str] = []

    if context_songs:
        song_list = "\n".join(f"- {song}" for song in context_songs)
        context_blocks.append(
            "The user has this playlist as reference:\n"
            f"{song_list}\n\nUse it to infer the style, mood, and genre. "
            "Recommend songs with the same vibe, but never include these songs."
        )

    if on_repeat_songs:
        taste_list = "\n".join(
            f"- {song['title']} by {song['artist']}" for song in on_repeat_songs[:30]
        )
        context_blocks.append(
            "The user's current favorite/most-played songs are:\n"
            f"{taste_list}\n\nUse these as their taste profile. Align with this taste, "
            "but never include these songs."
        )

    context_blocks.append(f"User request: {prompt}")
    result = await generate_structured(
        model=settings.openai_curation_model,
        schema_name="discover_playlist",
        schema=DISCOVER_SCHEMA,
        instructions=SYSTEM_PROMPT_WITH_PLAYLIST if save_to_playlist else SYSTEM_PROMPT,
        input_text="\n\n".join(context_blocks),
        max_output_tokens=8192,
        reasoning_effort="low",
        retries=1,
    )
    logger.info("[Discover] OpenAI returned %s songs", len(result["songs"]))
    return result


def _pick_best_track(items: list[dict]) -> dict | None:
    """From a list of Spotify track items, prefer the explicit version."""
    if not items:
        return None
    for track in items:
        if track.get("explicit", False):
            return track
    return items[0]


async def search_spotify(query: str, spotify_token: str) -> dict | None:
    """Search Spotify for a track. Prefers explicit versions."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            SPOTIFY_SEARCH_URL,
            params={"q": query, "type": "track", "limit": 10},
            headers={"Authorization": f"Bearer {spotify_token}"},
        )

    if resp.status_code != 200:
        logger.warning(
            "[Discover] Spotify search failed for '%s': status=%s", query, resp.status_code
        )
        return None

    items = resp.json().get("tracks", {}).get("items", [])
    track = _pick_best_track(items)
    if not track:
        return None

    album_images = track.get("album", {}).get("images", [])
    return {
        "title": track["name"],
        "artist": ", ".join(artist["name"] for artist in track["artists"]),
        "spotify_url": track["external_urls"].get("spotify"),
        "album_image": album_images[0]["url"] if album_images else None,
        "preview_url": track.get("preview_url"),
        "spotify_uri": track.get("uri"),
    }


async def validate_spotify_matches(
    original_prompt: str,
    recommendations: list[dict],
    spotify_results: list[dict],
) -> list[dict]:
    """Use GPT-6 Luna for a best-effort Spotify result quality check."""
    comparisons = []
    for index, (recommended, found) in enumerate(zip(recommendations, spotify_results)):
        if not found.get("spotify_uri"):
            continue
        comparisons.append(
            {
                "index": index,
                "requested": f"{recommended['title']} - {recommended['artist']}",
                "found": f"{found['title']} - {found['artist']}",
                "spotify_uri": found["spotify_uri"],
            }
        )

    if not comparisons:
        return spotify_results

    comparison_text = "\n".join(
        f"{comparison['index'] + 1}. Requested: \"{comparison['requested']}\" → "
        f"Found: \"{comparison['found']}\""
        for comparison in comparisons
    )
    instructions = """You are a music expert doing quality assurance on a playlist.
Compare each requested song with the Spotify result. Mark an item wrong only if it is a different song, a cover by another artist, or unrelated. Slight title variations such as Remastered or Live are acceptable. Be strict when artists differ completely."""
    input_text = (
        f'Original user request: "{original_prompt}"\n\n'
        f"Requested songs and Spotify results:\n{comparison_text}"
    )

    try:
        logger.info("[Discover QA] Validating %s Spotify matches", len(comparisons))
        result = await generate_structured(
            model=settings.openai_utility_model,
            schema_name="spotify_match_qa",
            schema=WRONG_INDICES_SCHEMA,
            instructions=instructions,
            input_text=input_text,
            max_output_tokens=1024,
            reasoning_effort="none",
            retries=0,
        )
        wrong_indices = set(result["wrong_indices"])
    except Exception as error:
        logger.warning("[Discover QA] Validation failed; retaining all results: %s", error)
        return spotify_results

    if not wrong_indices:
        logger.info("[Discover QA] All %s matches validated", len(comparisons))
        return spotify_results

    uris_to_remove = {
        comparison["spotify_uri"]
        for comparison in comparisons
        if comparison["index"] + 1 in wrong_indices
    }
    filtered = [
        song for song in spotify_results if song.get("spotify_uri") not in uris_to_remove
    ]
    logger.info(
        "[Discover QA] Kept %s/%s songs after filtering", len(filtered), len(spotify_results)
    )
    return filtered


async def discover_songs(
    prompt: str,
    spotify_token: str,
    context_songs: list[str] | None = None,
    on_repeat_songs: list[dict] | None = None,
    save_to_playlist: bool = False,
) -> dict:
    """Generate song recommendations, resolve them on Spotify, and validate matches."""
    logger.info("[Discover] Starting recommendation pipeline")
    result = await ask_openai(prompt, context_songs, on_repeat_songs, save_to_playlist)

    async def fetch_song(song: dict) -> dict:
        spotify_data = await search_spotify(f"{song['title']} {song['artist']}", spotify_token)
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

    songs = await asyncio.gather(*(fetch_song(song) for song in result["songs"]))
    logger.info(
        "[Discover] Spotify found %s/%s songs",
        sum(bool(song.get("spotify_uri")) for song in songs),
        len(songs),
    )
    songs = await validate_spotify_matches(prompt, result["songs"], list(songs))

    seen_uris: set[str] = set()
    unique_songs = []
    for song in songs:
        uri = song.get("spotify_uri")
        if uri and uri in seen_uris:
            continue
        if uri:
            seen_uris.add(uri)
        unique_songs.append(song)

    return {
        "mood_summary": result["mood_summary"],
        "playlist_name": result["playlist_name"],
        "playlist_description": result["playlist_description"],
        "songs": unique_songs,
    }
