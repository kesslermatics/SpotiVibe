"""
Gym Playlist generator – personalized workout playlist.

Flow:
1. User selects source playlists as inspiration
2. Fetch tracks from those playlists
3. Sample up to 15 inspiration tracks
4. Ask Gemini to generate 30 high-energy gym songs based on the user's taste
5. Search each song on Spotify (with Redis cache)
6. Delete old gym playlist if it exists
7. Create a new Spotify playlist with a unique date-based name
8. Optionally: auto-refresh daily at 3 AM (scheduler in main.py)
"""

import json
import random
import asyncio
import logging
from datetime import date

import httpx
import redis
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal
from app.models import User, GymPlaylistSettings
from app.auth import get_valid_spotify_token, refresh_spotify_token

settings = get_settings()
logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.1-pro-preview:generateContent?key={settings.gemini_api_key}"
)

# Redis Client
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


HISTORY_TTL = 2 * 24 * 60 * 60  # 2 days in seconds


def song_cache_key(title: str, artist: str) -> str:
    return f"song_uri::{title.lower().strip()}|||{artist.lower().strip()}"


def gym_history_key(user_id: int) -> str:
    return f"gym_history::{user_id}"


def save_gym_history(user_id: int, songs: list[dict]) -> None:
    """Save generated songs to Redis so they can be avoided next time.
    Songs expire after 2 days automatically via TTL."""
    key = gym_history_key(user_id)
    try:
        entries = [f"{s['title']} - {s['artist']}" for s in songs]
        # Append to existing history (don't overwrite)
        existing = redis_client.lrange(key, 0, -1)
        for entry in entries:
            if entry not in existing:
                redis_client.rpush(key, entry)
        # Reset TTL to 2 days from now
        redis_client.expire(key, HISTORY_TTL)
        logger.info(f"Gym History: Saved {len(entries)} songs for user {user_id} (total: {redis_client.llen(key)})")
    except Exception as e:
        logger.warning(f"Gym History: Could not save history: {e}")


def get_gym_history(user_id: int) -> list[str]:
    """Get recently used gym songs from Redis (last 2 days)."""
    key = gym_history_key(user_id)
    try:
        history = redis_client.lrange(key, 0, -1)
        logger.info(f"Gym History: Found {len(history)} recent songs for user {user_id}")
        return history
    except Exception as e:
        logger.warning(f"Gym History: Could not read history: {e}")
        return []


# ── Spotify helpers ───────────────────────────────────


async def fetch_playlist_tracks(
    playlist_id: str, spotify_token: str, user: User | None = None, db: Session | None = None
) -> tuple[list[dict], str]:
    """Fetch all tracks from a Spotify playlist.
    Returns (tracks, possibly_refreshed_token)."""
    tracks: list[dict] = []
    url = f"{SPOTIFY_API}/playlists/{playlist_id}/items"
    params: dict | None = {"limit": 50}
    headers = {"Authorization": f"Bearer {spotify_token}"}
    current_token = spotify_token

    async with httpx.AsyncClient(timeout=60) as client:
        while url:
            print(f"[GYM DEBUG] Fetching {url} with params={params}")
            resp = await client.get(url, params=params, headers=headers)
            print(f"[GYM DEBUG] fetch_playlist_tracks({playlist_id}): status={resp.status_code}")

            # Handle 401/403 by refreshing the token once
            if resp.status_code in (401, 403) and user and db:
                print(
                    f"[GYM DEBUG] fetch_playlist_tracks({playlist_id}): got {resp.status_code}, "
                    f"refreshing token... Response: {resp.text[:300]}"
                )
                try:
                    current_token = await refresh_spotify_token(user, db)
                    headers = {"Authorization": f"Bearer {current_token}"}
                    resp = await client.get(url, params=params, headers=headers)
                    print(
                        f"[GYM DEBUG] fetch_playlist_tracks({playlist_id}): after refresh status={resp.status_code} "
                        f"Response: {resp.text[:300]}"
                    )
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}")

            if resp.status_code == 403:
                print(
                    f"[GYM DEBUG] fetch_playlist_tracks({playlist_id}): 403 Forbidden! "
                    f"Response: {resp.text[:500]}"
                )
                return tracks, current_token

            if resp.status_code != 200:
                print(
                    f"[GYM DEBUG] fetch_playlist_tracks({playlist_id}): "
                    f"UNEXPECTED status {resp.status_code}: {resp.text[:500]}"
                )
                raise Exception(
                    f"Spotify error loading playlist {playlist_id}: "
                    f"HTTP {resp.status_code}"
                )

            data = resp.json()
            items = data.get("items", [])
            print(f"[GYM DEBUG] fetch_playlist_tracks({playlist_id}): got {len(items)} items in this page")

            for idx, item in enumerate(items):
                if idx == 0:
                    print(f"[GYM DEBUG] First item keys: {list(item.keys()) if isinstance(item, dict) else type(item)}")
                    print(f"[GYM DEBUG] First item sample: {str(item)[:500]}")
                track = item.get("track") or item.get("item")
                if not track:
                    if idx < 3:
                        print(f"[GYM DEBUG] Item {idx} has no 'track' key. Item: {str(item)[:300]}")
                    continue
                name = track.get("name")
                if not name:
                    if idx < 3:
                        print(f"[GYM DEBUG] Item {idx} track has no 'name'. Track: {str(track)[:300]}")
                    continue
                artists = track.get("artists", [])
                artist_name = ", ".join(
                    a["name"] for a in artists if a.get("name")
                ) if artists else "Unknown"
                tracks.append({
                    "title": name,
                    "artist": artist_name,
                    "uri": track.get("uri", ""),
                })

            url = data.get("next")
            params = None  # next URL already includes all params

    print(f"[GYM DEBUG] fetch_playlist_tracks({playlist_id}): TOTAL {len(tracks)} tracks")
    return tracks, current_token


async def robust_spotify_search(
    query: str, spotify_token: str, max_retries: int = 3
) -> dict | None:
    """Spotify search with retry-after handling."""
    headers = {"Authorization": f"Bearer {spotify_token}"}
    params = {"q": query, "type": "track", "limit": 1}

    for attempt in range(max_retries):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{SPOTIFY_API}/search", params=params, headers=headers
            )
        if resp.status_code == 200:
            items = resp.json().get("tracks", {}).get("items", [])
            if not items:
                return None
            track = items[0]
            return {
                "title": track["name"],
                "artist": ", ".join(a["name"] for a in track["artists"]),
                "uri": track["uri"],
            }
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "3")
            wait = min(int(retry_after), 30) if retry_after.isdigit() else 3
            logger.warning(
                f"Spotify search 429 for '{query}', waiting {wait}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(wait)
            continue
        logger.warning(f"Spotify search failed for '{query}': {resp.status_code}")
        return None

    return None


async def robust_spotify_search_with_cache(
    title: str, artist: str, spotify_token: str
) -> dict | None:
    """Search Spotify with Redis cache."""
    key = song_cache_key(title, artist)
    try:
        cached = redis_client.get(key)
        if cached and cached.startswith("spotify:track:"):
            return {"title": title, "artist": artist, "uri": cached}
    except Exception:
        pass

    result = await robust_spotify_search(f"{title} {artist}", spotify_token)
    if result and result.get("uri"):
        try:
            redis_client.set(key, result["uri"])
        except Exception:
            pass
        return result
    return None


async def delete_spotify_playlist(
    playlist_id: str, spotify_token: str
) -> bool:
    """Unfollow (delete) a Spotify playlist. Returns True on success."""
    async with httpx.AsyncClient() as client:
        resp = await client.delete(
            f"{SPOTIFY_API}/playlists/{playlist_id}/followers",
            headers={"Authorization": f"Bearer {spotify_token}"},
        )
    if resp.status_code == 200:
        logger.info(f"Deleted old gym playlist {playlist_id}")
        return True
    logger.warning(
        f"Failed to delete playlist {playlist_id}: {resp.status_code} {resp.text[:200]}"
    )
    return False


async def robust_add_items(
    client: httpx.AsyncClient,
    playlist_id: str,
    uris: list[str],
    headers: dict,
    max_retries: int = 3,
) -> bool:
    """Add items to a playlist with retry logic."""
    for attempt in range(max_retries):
        resp = await client.post(
            f"{SPOTIFY_API}/playlists/{playlist_id}/tracks",
            headers=headers,
            json={"uris": uris},
        )
        if resp.status_code in (200, 201):
            return True
        if resp.status_code == 429:
            wait = min(int(resp.headers.get("Retry-After", "3")), 30)
            await asyncio.sleep(wait)
            continue
        logger.error(f"Failed to add tracks: {resp.status_code} {resp.text[:300]}")
        return False
    return False


# ── Gemini ────────────────────────────────────────────


async def ask_gemini_gym(inspiration_songs: list[str], recent_history: list[str] | None = None) -> dict:
    """Ask Gemini for a gym playlist based on inspiration songs.
    If recent_history is provided, Gemini will avoid those songs."""
    song_list = "\n".join(f"- {s}" for s in inspiration_songs)

    avoid_block = ""
    if recent_history:
        avoid_list = "\n".join(f"- {s}" for s in recent_history)
        avoid_block = f"""\n\nIMPORTANT: The following songs were already used in recent gym playlists (last 2 days).
DO NOT include ANY of these songs. Pick DIFFERENT songs instead:
{avoid_list}\n"""

    prompt = f"""You are a music expert creating personalized gym playlists.

I will give you a list of songs that represent the user's ACTUAL music taste.

CRITICAL: The user's taste is DIVERSE. Analyze the genres in their inspiration songs carefully.
If they listen to 60% Rock, 20% Pop, 20% Hip-Hop – your playlist should reflect similar proportions!
Do NOT convert everything into one genre (like hard metal). PRESERVE their genre diversity.

Your task: Create an energizing workout playlist with exactly 30 songs that:
- MATCH the user's actual genre distribution from the inspiration songs
- Pick energetic/upbeat songs FROM EACH GENRE the user likes (not just rock/metal)
- For Pop → pick danceable, high-BPM pop tracks  
- For Hip-Hop → pick hype, motivating rap tracks
- For Rock → pick energetic rock (but NOT just metal!)
- For Electronic → pick driving EDM/dance tracks
- Etc. for any other genres present
- Avoid slow ballads, but don't interpret "workout" as "must be metal"
- Mix well-known tracks with some discoveries
- Create variety – avoid 30 songs that sound identical

DO NOT include any of the inspiration songs in your recommendations.
{avoid_block}
Respond ONLY with valid JSON in this exact format:
{{
  "songs": [
    {{"title": "Song Name", "artist": "Artist Name"}},
    ...
  ]
}}

Rules:
- Exactly 30 songs
- No duplicates  
- Reflect the user's genre mix, not just one style
- Energetic tracks from EACH genre they like
- Only output valid JSON, no markdown, no explanation

Here are the user's inspiration songs (analyze genres carefully):
{song_list}"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": 8192,
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(GEMINI_URL, json=payload)

    if resp.status_code != 200:
        raise Exception(f"Gemini API error: {resp.status_code} – {resp.text[:300]}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected Gemini response: {e}")

    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Gemini returned invalid JSON: {text[:500]}")
        raise Exception(f"Gemini returned invalid JSON: {e}")


# ── Main generation pipeline ─────────────────────────


async def generate_gym_playlist(
    source_playlist_ids: list[str],
    current_user: User,
    db: Session,
) -> dict:
    """
    Full gym playlist generation pipeline.
    """
    spotify_token = await get_valid_spotify_token(current_user, db)

    # 1. Fetch tracks from all selected playlists
    logger.info(
        f"Gym Playlist: Fetching tracks from {len(source_playlist_ids)} playlists..."
    )
    all_tracks: list[dict] = []
    skipped_playlists: list[str] = []
    for pid in source_playlist_ids:
        try:
            tracks, spotify_token = await fetch_playlist_tracks(
                pid, spotify_token, user=current_user, db=db
            )
            if tracks:
                all_tracks.extend(tracks)
            else:
                skipped_playlists.append(pid)
                logger.warning(f"Gym Playlist: Playlist {pid} returned 0 tracks (skipped)")
        except Exception as e:
            skipped_playlists.append(pid)
            logger.warning(f"Gym Playlist: Skipping playlist {pid} due to error: {e}")
        if len(source_playlist_ids) > 1:
            await asyncio.sleep(0.3)

    if skipped_playlists:
        logger.info(
            f"Gym Playlist: Skipped {len(skipped_playlists)}/{len(source_playlist_ids)} "
            f"playlists (403/inaccessible): {skipped_playlists}"
        )

    if len(all_tracks) < 5:
        raise Exception(
            "Too few songs in the selected playlists. "
            "Some playlists could not be loaded (e.g. Spotify-generated ones like 'Discover Weekly'). "
            "Choose other playlists with more of your own songs!"
        )

    logger.info(f"Gym Playlist: Got {len(all_tracks)} total tracks")

    # 2. Sample inspiration songs (up to 15)
    sample_size = min(15, len(all_tracks))
    sampled = random.sample(all_tracks, sample_size)
    inspiration = [f"{t['title']} - {t['artist']}" for t in sampled]
    logger.info(f"Gym Playlist: Using {len(inspiration)} inspiration songs")

    # 3. Load recent song history & ask Gemini
    recent_history = get_gym_history(current_user.id)
    logger.info(f"Gym Playlist: {len(recent_history)} songs in 2-day history to avoid")

    logger.info("Gym Playlist: Asking Gemini for recommendations...")
    gemini_result = await ask_gemini_gym(inspiration, recent_history if recent_history else None)
    gemini_songs = gemini_result.get("songs", [])
    logger.info(f"Gym Playlist: Gemini returned {len(gemini_songs)} songs")

    # 4. Search each song on Spotify (sequential with delay)
    logger.info("Gym Playlist: Searching songs on Spotify...")
    uris: list[str] = []
    seen_uris: set[str] = set()
    for song in gemini_songs:
        result = await robust_spotify_search_with_cache(
            song["title"], song["artist"], spotify_token
        )
        if result and result.get("uri"):
            uri = result["uri"]
            if uri not in seen_uris:
                uris.append(uri)
                seen_uris.add(uri)
        await asyncio.sleep(1.0)

    logger.info(f"Gym Playlist: Found {len(uris)} tracks on Spotify")

    # 4b. Save generated songs to history (2-day TTL)
    save_gym_history(current_user.id, gemini_songs)

    if len(uris) < 10:
        raise Exception(
            "Too few songs found on Spotify. Please try again!"
        )

    # 5. Delete old gym playlist if it exists
    try:
        gym_settings = (
            db.query(GymPlaylistSettings)
            .filter(GymPlaylistSettings.user_id == current_user.id)
            .first()
        )
    except Exception as e:
        logger.warning(f"Gym Playlist: Could not query GymPlaylistSettings (table may not exist): {e}")
        gym_settings = None

    if gym_settings and gym_settings.last_spotify_playlist_id:
        logger.info(
            f"Gym Playlist: Deleting old playlist {gym_settings.last_spotify_playlist_id}"
        )
        await delete_spotify_playlist(
            gym_settings.last_spotify_playlist_id, spotify_token
        )

    # 6. Create new playlist with unique name
    today = date.today().strftime("%d.%m.%Y")
    playlist_name = f"🏋️ VibeSwipe Gym Mix – {today}"
    playlist_desc = (
        f"Your personal Gym Power Mix by VibeSwipe 💪 "
        f"{len(uris)} motivating tracks"
    )

    # Refresh token before playlist operations (may have gone stale during searches)
    spotify_token = await get_valid_spotify_token(current_user, db)
    auth_headers = {"Authorization": f"Bearer {spotify_token}"}

    # 6a. Create playlist (eigener Client – wie in routes.py)
    async with httpx.AsyncClient() as client:
        create_resp = await client.post(
            f"{SPOTIFY_API}/me/playlists",
            headers=auth_headers,
            json={
                "name": playlist_name,
                "description": playlist_desc,
                "public": False,
            },
        )

    if create_resp.status_code not in (200, 201):
        logger.error(
            f"Gym Playlist: Playlist creation failed: {create_resp.status_code} {create_resp.text[:500]}"
        )
        raise Exception(
            f"Could not create playlist: HTTP {create_resp.status_code} – {create_resp.text[:300]}"
        )

    playlist = create_resp.json()
    playlist_id = playlist["id"]
    print(f"[GYM DEBUG] Playlist created: {playlist_id}")

    # Small delay to let Spotify propagate the new playlist
    await asyncio.sleep(1)

    # 6b. Add tracks in chunks of 100 (eigener Client pro Chunk – wie in routes.py)
    for i in range(0, len(uris), 100):
        chunk = uris[i : i + 100]
        async with httpx.AsyncClient() as client:
            add_resp = await client.post(
                f"{SPOTIFY_API}/playlists/{playlist_id}/items",
                headers=auth_headers,
                json={"uris": chunk},
            )
        if add_resp.status_code not in (200, 201):
            print(
                f"[GYM DEBUG] Failed to add tracks chunk {i}: "
                f"status={add_resp.status_code} body={add_resp.text[:500]}"
            )
            logger.error(f"Failed to add tracks chunk {i}: {add_resp.status_code}")
        else:
            print(f"[GYM DEBUG] Added chunk {i} ({len(chunk)} tracks) to playlist")

    # 7. Save/update settings in DB
    auto_refresh_val = False
    try:
        if not gym_settings:
            gym_settings = GymPlaylistSettings(
                user_id=current_user.id,
                source_playlist_ids=json.dumps(source_playlist_ids),
                last_spotify_playlist_id=playlist_id,
                auto_refresh=False,
            )
            db.add(gym_settings)
        else:
            gym_settings.source_playlist_ids = json.dumps(source_playlist_ids)
            gym_settings.last_spotify_playlist_id = playlist_id
            auto_refresh_val = gym_settings.auto_refresh

        db.commit()
    except Exception as e:
        logger.warning(f"Gym Playlist: Could not save settings to DB: {e}")
        db.rollback()

    return {
        "playlist_url": playlist["external_urls"]["spotify"],
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "total_tracks": len(uris),
        "inspiration_count": len(inspiration),
        "auto_refresh": auto_refresh_val,
    }


# ── Auto-refresh job (called by scheduler) ───────────


async def auto_refresh_gym_playlists():
    """
    Scheduled job: regenerate gym playlists for all users with auto_refresh=True.
    Called daily at 3:00 AM.
    """
    logger.info("Gym Playlist Auto-Refresh: Starting...")

    db = SessionLocal()
    try:
        settings_list = (
            db.query(GymPlaylistSettings)
            .filter(GymPlaylistSettings.auto_refresh == True)  # noqa: E712
            .all()
        )
        logger.info(
            f"Gym Playlist Auto-Refresh: Found {len(settings_list)} users with auto-refresh"
        )

        for gym_settings in settings_list:
            try:
                user = db.query(User).filter(User.id == gym_settings.user_id).first()
                if not user:
                    logger.warning(
                        f"Auto-Refresh: User {gym_settings.user_id} not found, skipping"
                    )
                    continue

                source_ids = json.loads(gym_settings.source_playlist_ids or "[]")
                if not source_ids:
                    logger.warning(
                        f"Auto-Refresh: User {user.spotify_id} has no source playlists, skipping"
                    )
                    continue

                logger.info(
                    f"Auto-Refresh: Generating gym playlist for user {user.spotify_id}..."
                )
                await generate_gym_playlist(source_ids, user, db)
                logger.info(f"Auto-Refresh: Success for user {user.spotify_id}")

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(
                    f"Auto-Refresh: Failed for user {gym_settings.user_id}: {e}",
                    exc_info=True,
                )
                continue

    finally:
        db.close()

    logger.info("Gym Playlist Auto-Refresh: Done.")
