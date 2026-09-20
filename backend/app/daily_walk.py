"""
Daily Walk generator – a chill mix of music & podcasts for your daily walk.

Like Daily Drive, but with a different Songs/Podcast ratio:
  Daily Drive: 4 songs → 1 podcast episode
  Daily Walk:  2 songs → 1 podcast episode  (more podcasts, less music)

Supports auto_refresh: settings are persisted per user so the scheduler
can regenerate the playlist every day at 4:00 AM (like Gym Playlist does at 3:00).

Flow:
1. Fetch user's top tracks (short_term = "On Repeat") from Spotify
2. Gemini curates a song selection (familiar + discoveries)
3. Fetch podcast episodes from the user's saved shows selection
4. Interleave: 2 songs → 1 episode → 2 songs → 1 episode …
5. Create / overwrite the Spotify playlist
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
from app.models import User, DailyWalkSettings
from app.database import SessionLocal
from app.auth import get_valid_spotify_token

settings = get_settings()
logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.6-flash:generateContent?key={settings.gemini_api_key}"
)

# Shared Redis client
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

# History TTL: 5 days
HISTORY_TTL = 5 * 24 * 60 * 60

# Songs-per-episode ratio for Daily Walk (2 songs then 1 episode)
SONGS_PER_EPISODE = 2


# ── Redis helpers ─────────────────────────────────────

def song_cache_key(title: str, artist: str) -> str:
    return f"song_uri_v2::{title.lower().strip()}|||{artist.lower().strip()}"


def daily_walk_history_key(user_id: int) -> str:
    return f"daily_walk_history::{user_id}"


def save_daily_walk_history(user_id: int, songs: list[dict]) -> None:
    key = daily_walk_history_key(user_id)
    try:
        existing_raw = redis_client.get(key)
        existing: list[str] = json.loads(existing_raw) if existing_raw else []
        new_entries = [f"{s['title']} – {s['artist']}" for s in songs]
        combined = new_entries + existing
        # Keep max 200 entries to avoid prompt bloat
        combined = combined[:200]
        redis_client.setex(key, HISTORY_TTL, json.dumps(combined))
    except Exception as e:
        logger.warning(f"Daily Walk: Could not save history to Redis: {e}")


def get_daily_walk_history(user_id: int) -> list[str]:
    key = daily_walk_history_key(user_id)
    try:
        raw = redis_client.get(key)
        return json.loads(raw) if raw else []
    except Exception:
        return []


# ── Spotify helpers (shared logic from daily_drive) ───

async def fetch_on_repeat_tracks(spotify_token: str) -> list[dict]:
    """Fetch user's short-term top tracks ("on repeat")."""
    headers = {"Authorization": f"Bearer {spotify_token}"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{SPOTIFY_API}/me/top/tracks",
            params={"time_range": "short_term", "limit": 50},
            headers=headers,
        )
    if resp.status_code != 200:
        logger.error(f"Daily Walk: Spotify top tracks error {resp.status_code}: {resp.text[:200]}")
        raise Exception(f"Could not fetch top tracks: {resp.status_code}")

    items = resp.json().get("items", [])
    tracks = []
    for t in items:
        tracks.append({
            "title": t["name"],
            "artist": ", ".join(a["name"] for a in t["artists"]),
            "uri": t["uri"],
            "id": t["id"],
        })
    return tracks


async def fetch_saved_shows(spotify_token: str) -> list[dict]:
    """Fetch user's followed/saved podcasts."""
    headers = {"Authorization": f"Bearer {spotify_token}"}
    shows = []
    url = f"{SPOTIFY_API}/me/shows"
    params = {"limit": 50}

    async with httpx.AsyncClient(timeout=30) as client:
        while url:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                break
            data = resp.json()
            for item in data.get("items", []):
                show = item.get("show", {})
                images = show.get("images", [])
                shows.append({
                    "id": show["id"],
                    "name": show["name"],
                    "publisher": show.get("publisher", ""),
                    "image": images[0]["url"] if images else None,
                    "total_episodes": show.get("total_episodes", 0),
                })
            url = data.get("next")
            params = {}  # next URL already contains params

    return shows


async def fetch_show_episodes(show_id: str, spotify_token: str, limit: int = 20) -> list[dict]:
    """Fetch recent episodes from a podcast show."""
    headers = {"Authorization": f"Bearer {spotify_token}"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{SPOTIFY_API}/shows/{show_id}/episodes",
            params={"limit": limit, "market": "DE"},
            headers=headers,
        )
    if resp.status_code != 200:
        logger.warning(f"Daily Walk: Could not fetch episodes for show {show_id}: {resp.status_code}")
        return []

    episodes = []
    for ep in resp.json().get("items", []):
        resume = ep.get("resume_point", {})
        episodes.append({
            "id": ep["id"],
            "name": ep["name"],
            "uri": ep["uri"],
            "release_date": ep.get("release_date", ""),
            "fully_played": resume.get("fully_played", False),
        })
    return episodes


# ── Gemini curation ───────────────────────────────────

async def ask_gemini_daily_walk(
    on_repeat_songs: list[dict],
    duration_minutes: int,
    walk_mood: str,
    familiarity: int,
    recent_history: list[str] | None = None,
) -> dict:
    """Ask Gemini to curate songs for a Daily Walk playlist."""
    song_list = "\n".join(
        f"- {s['title']} – {s['artist']}" for s in on_repeat_songs
    )

    # Daily Walk is more relaxed – shorter chunks of music between podcast episodes.
    # With 2 songs per episode, we need fewer songs overall.
    target_song_count = max(4, min(40, round(duration_minutes / 4.0)))
    num_new = round(target_song_count * familiarity / 100)
    num_from_repeat = target_song_count - num_new

    mood_guidance = {
        "chill": "Create a relaxed, easy-going atmosphere: calm energy, clear headspace, great for a peaceful morning or afternoon stroll.",
        "energetic": "Create an upbeat, motivating vibe: moderate-high energy, feel-good beats, the kind of music that puts a spring in your step.",
        "focus": "Create a focused, deep-listening experience: atmospheric, minimal distraction, good for a contemplative solo walk.",
    }.get(walk_mood, "Create a balanced, versatile mix suitable for a casual walk at any time of day.")

    avoid_block = ""
    if recent_history:
        avoid_list = "\n".join(f"- {s}" for s in recent_history[:150])
        avoid_block = f"""

CRITICAL – AVOID REPEATS: The following songs were used in recent Daily Walk playlists.
Do NOT include ANY of these songs. Pick COMPLETELY DIFFERENT ones instead:
{avoid_list}
"""

    seed_hint = random.randint(1000, 9999)

    prompt = f"""You are a music curation expert building a "Daily Walk" playlist.
Today's session seed: {seed_hint} (use this to vary your picks!)

The user goes for a walk and wants a relaxed mix of music interrupted regularly by podcast episodes.
Music should complement, not overpower – think quality over quantity.

Your task:
1. Pick exactly {num_from_repeat} songs FROM the provided on-repeat list. Use EXACT titles and artists.
2. Recommend exactly {num_new} NEW songs not in the list, matching the style and mood.

Walk duration: approximately {duration_minutes} minutes (music fills the gaps between podcast episodes).
Mood: {walk_mood}. {mood_guidance}
Discovery setting: {familiarity}% new → {num_from_repeat} familiar + {num_new} new.
{avoid_block}
Respond ONLY with valid JSON in this exact format, nothing else:
{{
  "from_repeat": [
    {{"title": "Song Name", "artist": "Artist Name"}},
    ...
  ],
  "new_discoveries": [
    {{"title": "Song Name", "artist": "Artist Name"}},
    ...
  ]
}}

Rules:
- "from_repeat" must contain exactly {num_from_repeat} songs FROM the provided list (exact titles/artists)
- "new_discoveries" must contain exactly {num_new} songs NOT in the list
- Keep the energy relaxed and walk-friendly – avoid very aggressive or jarring tracks
- Mix eras, sub-genres, and tempos for a pleasant journey
- No duplicates
- Only output valid JSON, no markdown, no explanation"""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"text": f"Here are the user's On-Repeat songs:\n{song_list}"},
            ]
        }],
        "generationConfig": {
            "temperature": 1.6,
            "maxOutputTokens": 4096,
            "topP": 0.95,
            "topK": 64,
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(GEMINI_URL, json=payload)

    if resp.status_code != 200:
        logger.error(f"Daily Walk Gemini error: {resp.status_code} – {resp.text[:500]}")
        raise Exception(f"Gemini API error: {resp.status_code}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error(f"Daily Walk: Unexpected Gemini response: {json.dumps(data)[:500]}")
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
        logger.error(f"Daily Walk: Gemini invalid JSON: {text[:500]}")
        raise Exception(f"Gemini returned invalid JSON: {e}")


# ── Spotify search helpers ─────────────────────────────

def _pick_best_track(items: list[dict]) -> dict | None:
    if not items:
        return None
    for track in items:
        if track.get("explicit", False):
            return track
    return items[0]


async def robust_spotify_search(query: str, spotify_token: str, max_retries: int = 3) -> dict | None:
    headers = {"Authorization": f"Bearer {spotify_token}"}
    params = {"q": query, "type": "track", "limit": 10}
    for attempt in range(max_retries):
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{SPOTIFY_API}/search", params=params, headers=headers)
        if resp.status_code == 200:
            items = resp.json().get("tracks", {}).get("items", [])
            track = _pick_best_track(items)
            if not track:
                return None
            return {
                "title": track["name"],
                "artist": ", ".join(a["name"] for a in track["artists"]),
                "uri": track["uri"],
                "id": track["id"],
            }
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            logger.warning(f"Daily Walk: Spotify 429, waiting {retry_after}s (attempt {attempt+1})")
            await asyncio.sleep(retry_after)
        else:
            logger.warning(f"Daily Walk: Spotify search {resp.status_code} for '{query}'")
            return None
    return None


async def robust_spotify_search_with_cache(title: str, artist: str, spotify_token: str, max_retries: int = 3) -> dict | None:
    cache_key = song_cache_key(title, artist)
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    query = f"{title} {artist}"
    result = await robust_spotify_search(query, spotify_token, max_retries)

    if result:
        try:
            redis_client.setex(cache_key, 7 * 24 * 3600, json.dumps(result))
        except Exception:
            pass

    return result


async def robust_add_items_to_playlist(
    client: httpx.AsyncClient,
    playlist_id: str,
    chunk: list[str],
    auth_headers: dict,
    max_retries: int = 3,
) -> bool:
    for attempt in range(max_retries):
        resp = await client.post(
            f"{SPOTIFY_API}/playlists/{playlist_id}/tracks",
            headers=auth_headers,
            json={"uris": chunk},
        )
        if resp.status_code in (200, 201):
            return True
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            logger.warning(f"Daily Walk: Playlist add 429, waiting {retry_after}s")
            await asyncio.sleep(retry_after)
        else:
            logger.error(f"Daily Walk: Add items failed {resp.status_code}: {resp.text[:200]}")
            return False
    return False


# ── Main generator ────────────────────────────────────

async def generate_daily_walk(
    spotify_token: str,
    spotify_user_id: str,
    selected_show_ids: list[str],
    duration_minutes: int,
    walk_mood: str,
    familiarity: int,
    user_id: int | None = None,
    existing_playlist_id: str | None = None,
) -> dict:
    """
    Full Daily Walk generation pipeline.
    If existing_playlist_id is given, the old playlist is replaced (items cleared, then refilled).
    Returns info about the created/updated playlist.
    """
    # 1. Fetch On-Repeat tracks
    logger.info("Daily Walk: Fetching On-Repeat tracks...")
    on_repeat = await fetch_on_repeat_tracks(spotify_token)
    logger.info(f"Daily Walk: Got {len(on_repeat)} On-Repeat tracks")
    if len(on_repeat) < 5:
        raise Exception(
            "You need at least 5 songs in your Top Tracks to use Daily Walk. "
            "Listen to more music and try again!"
        )

    # 2. Fetch podcast episodes early (while Gemini processes)
    unplayed_episodes: list[dict] = []
    played_episodes: list[dict] = []
    if selected_show_ids:
        logger.info(f"Daily Walk: Fetching episodes for {len(selected_show_ids)} show(s)...")
        for show_id in selected_show_ids:
            eps = await fetch_show_episodes(show_id, spotify_token, limit=20)
            for ep in eps:
                if ep["fully_played"]:
                    played_episodes.append(ep)
                else:
                    unplayed_episodes.append(ep)
            if len(selected_show_ids) > 1:
                await asyncio.sleep(0.5)
        logger.info(f"Daily Walk: {len(unplayed_episodes)} unplayed + {len(played_episodes)} played episodes")

    # 3. Gemini curation
    recent_history = get_daily_walk_history(user_id) if user_id else []
    logger.info(f"Daily Walk: {len(recent_history)} songs in history to avoid")
    logger.info("Daily Walk: Asking Gemini to curate songs...")
    gemini_result = await ask_gemini_daily_walk(
        on_repeat,
        duration_minutes=duration_minutes,
        walk_mood=walk_mood,
        familiarity=familiarity,
        recent_history=recent_history if recent_history else None,
    )
    logger.info(
        f"Daily Walk: Gemini returned {len(gemini_result.get('from_repeat', []))} from_repeat, "
        f"{len(gemini_result.get('new_discoveries', []))} new_discoveries"
    )

    # 4. Map from_repeat back to Spotify URIs
    on_repeat_map: dict[str, dict] = {}
    for t in on_repeat:
        key = f"{t['title'].lower().strip()}|||{t['artist'].lower().strip()}"
        on_repeat_map[key] = t
        title_key = f"title:::{t['title'].lower().strip()}"
        if title_key not in on_repeat_map:
            on_repeat_map[title_key] = t

    from_repeat_uris: list[str] = []
    unmatched_from_repeat: list[dict] = []
    for song in gemini_result.get("from_repeat", []):
        key = f"{song['title'].lower().strip()}|||{song['artist'].lower().strip()}"
        if key in on_repeat_map:
            from_repeat_uris.append(on_repeat_map[key]["uri"])
        else:
            title_key = f"title:::{song['title'].lower().strip()}"
            if title_key in on_repeat_map:
                from_repeat_uris.append(on_repeat_map[title_key]["uri"])
            else:
                unmatched_from_repeat.append(song)

    # 5. Search Spotify for unmatched + new discoveries
    all_to_search = (
        [{"song": s, "type": "from_repeat"} for s in unmatched_from_repeat]
        + [{"song": s, "type": "new_discovery"} for s in gemini_result.get("new_discoveries", [])]
    )
    logger.info(f"Daily Walk: Searching {len(all_to_search)} songs on Spotify...")

    new_discovery_uris: list[str] = []
    for item in all_to_search:
        song = item["song"]
        search_result = await robust_spotify_search_with_cache(song["title"], song["artist"], spotify_token)
        uri = search_result["uri"] if search_result else None
        if uri:
            if item["type"] == "from_repeat":
                from_repeat_uris.append(uri)
            else:
                new_discovery_uris.append(uri)
        await asyncio.sleep(1.2)

    # 5b. Deduplicate
    seen_uris: set[str] = set()
    from_repeat_uris_deduped: list[str] = []
    for uri in from_repeat_uris:
        if uri not in seen_uris:
            from_repeat_uris_deduped.append(uri)
            seen_uris.add(uri)
    new_discovery_uris_deduped: list[str] = []
    for uri in new_discovery_uris:
        if uri not in seen_uris:
            new_discovery_uris_deduped.append(uri)
            seen_uris.add(uri)
    from_repeat_uris = from_repeat_uris_deduped
    new_discovery_uris = new_discovery_uris_deduped

    logger.info(f"Daily Walk: {len(from_repeat_uris)} from_repeat, {len(new_discovery_uris)} new discoveries (after dedup)")

    # 5c. Save to history
    if user_id:
        all_for_history = (
            list(gemini_result.get("from_repeat", []))
            + list(gemini_result.get("new_discoveries", []))
        )
        save_daily_walk_history(user_id, all_for_history)

    # 6. Shuffle and interleave familiar/new songs
    random.shuffle(from_repeat_uris)
    random.shuffle(new_discovery_uris)

    all_song_uris: list[str] = []
    repeat_iter = iter(from_repeat_uris)
    new_iter = iter(new_discovery_uris)
    use_repeat = True
    while True:
        src = repeat_iter if use_repeat else new_iter
        uri = next(src, None)
        if uri is None:
            other = new_iter if use_repeat else repeat_iter
            all_song_uris.extend(other)
            break
        all_song_uris.append(uri)
        use_repeat = not use_repeat

    # 7. Pick podcast episodes
    episode_uris: list[str] = []
    if selected_show_ids:
        unplayed_episodes.sort(key=lambda e: e.get("release_date", ""), reverse=True)
        played_episodes.sort(key=lambda e: e.get("release_date", ""), reverse=True)

        # With 2 songs per episode we need more episodes than Daily Drive
        needed = max(1, len(all_song_uris) // SONGS_PER_EPISODE)
        chosen_episodes = unplayed_episodes[:needed]
        if len(chosen_episodes) < needed:
            remaining = needed - len(chosen_episodes)
            chosen_episodes.extend(played_episodes[:remaining])

        logger.info(
            f"Daily Walk: Picked {len(chosen_episodes)} episodes "
            f"({min(needed, len(unplayed_episodes))} unplayed, "
            f"{max(0, len(chosen_episodes) - len(unplayed_episodes))} played fallback)"
        )
        episode_uris = [ep["uri"] for ep in chosen_episodes]

    # 8. Interleave: SONGS_PER_EPISODE songs → 1 episode → repeat
    final_uris: list[str] = []
    song_idx = 0
    ep_idx = 0
    while song_idx < len(all_song_uris):
        chunk = all_song_uris[song_idx: song_idx + SONGS_PER_EPISODE]
        final_uris.extend(chunk)
        song_idx += SONGS_PER_EPISODE
        if ep_idx < len(episode_uris):
            final_uris.append(episode_uris[ep_idx])
            ep_idx += 1

    # 9. Create or replace the Spotify playlist
    today = date.today().strftime("%d.%m.%Y")
    playlist_name = f"Daily Walk – {today}"
    playlist_desc = (
        f"Your personal Daily Walk by VibeSwipe 🚶 "
        f"{len(from_repeat_uris)} On-Repeat Songs, "
        f"{len(new_discovery_uris)} new discoveries"
        f"{f', {len(episode_uris)} podcast episodes' if episode_uris else ''}"
    )
    auth_headers = {"Authorization": f"Bearer {spotify_token}"}

    async with httpx.AsyncClient() as client:
        if existing_playlist_id:
            # Rename and clear existing playlist
            await client.put(
                f"{SPOTIFY_API}/playlists/{existing_playlist_id}",
                headers=auth_headers,
                json={"name": playlist_name, "description": playlist_desc},
            )
            # Remove all existing tracks
            current_tracks_resp = await client.get(
                f"{SPOTIFY_API}/playlists/{existing_playlist_id}/tracks",
                headers=auth_headers,
                params={"fields": "items(track(uri)),next", "limit": 100},
            )
            if current_tracks_resp.status_code == 200:
                existing_uris = [
                    item["track"]["uri"]
                    for item in current_tracks_resp.json().get("items", [])
                    if item.get("track")
                ]
                if existing_uris:
                    await client.request(
                        "DELETE",
                        f"{SPOTIFY_API}/playlists/{existing_playlist_id}/tracks",
                        headers=auth_headers,
                        json={"tracks": [{"uri": u} for u in existing_uris]},
                    )
            playlist_id = existing_playlist_id
            playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
            logger.info(f"Daily Walk: Reusing existing playlist {playlist_id}")
        else:
            # Create new playlist
            create_resp = await client.post(
                f"{SPOTIFY_API}/me/playlists",
                headers=auth_headers,
                json={"name": playlist_name, "description": playlist_desc, "public": False},
            )
            if create_resp.status_code not in (200, 201):
                raise Exception(f"Could not create playlist: {create_resp.text}")
            playlist = create_resp.json()
            playlist_id = playlist["id"]
            playlist_url = playlist["external_urls"]["spotify"]
            logger.info(f"Daily Walk: Created new playlist {playlist_id}")

        # Add tracks in chunks of 100
        for i in range(0, len(final_uris), 100):
            chunk = final_uris[i: i + 100]
            success = await robust_add_items_to_playlist(client, playlist_id, chunk, auth_headers)
            if not success:
                logger.error(f"Daily Walk: Failed to add chunk {i}-{i+len(chunk)} after retries")

    logger.info(f"Daily Walk: Done. Playlist {playlist_id} has {len(final_uris)} items.")
    return {
        "playlist_url": playlist_url,
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "total_tracks": len(final_uris),
        "on_repeat_count": len(from_repeat_uris),
        "new_discoveries_count": len(new_discovery_uris),
        "episodes_count": len(episode_uris),
    }


# ── Scheduler job ─────────────────────────────────────

async def auto_refresh_daily_walk_playlists() -> None:
    """
    Scheduled job: regenerate Daily Walk playlists for all users with auto_refresh=True.
    Called daily at 4:00 AM.
    """
    logger.info("Daily Walk Auto-Refresh: Starting...")

    db: Session = SessionLocal()
    try:
        settings_list = (
            db.query(DailyWalkSettings)
            .filter(DailyWalkSettings.auto_refresh == True)  # noqa: E712
            .all()
        )
        logger.info(f"Daily Walk Auto-Refresh: {len(settings_list)} user(s) with auto-refresh enabled")

        for walk_settings in settings_list:
            try:
                user = db.query(User).filter(User.id == walk_settings.user_id).first()
                if not user:
                    logger.warning(f"Daily Walk Auto-Refresh: User {walk_settings.user_id} not found, skipping")
                    continue

                # Refresh the Spotify token
                spotify_token = await get_valid_spotify_token(user, db)

                selected_show_ids: list[str] = []
                try:
                    selected_show_ids = json.loads(walk_settings.selected_show_ids or "[]")
                except Exception:
                    pass

                logger.info(f"Daily Walk Auto-Refresh: Generating for user {user.spotify_id}...")
                result = await generate_daily_walk(
                    spotify_token=spotify_token,
                    spotify_user_id=user.spotify_id,
                    selected_show_ids=selected_show_ids,
                    duration_minutes=walk_settings.duration_minutes,
                    walk_mood=walk_settings.walk_mood,
                    familiarity=walk_settings.familiarity,
                    user_id=user.id,
                    existing_playlist_id=walk_settings.last_spotify_playlist_id,
                )

                # Persist the new playlist ID back
                walk_settings.last_spotify_playlist_id = result["playlist_id"]
                db.commit()

                logger.info(f"Daily Walk Auto-Refresh: Success for user {user.spotify_id}")
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(
                    f"Daily Walk Auto-Refresh: Failed for user {walk_settings.user_id}: {e}",
                    exc_info=True,
                )
                continue

    finally:
        db.close()

    logger.info("Daily Walk Auto-Refresh: Done.")
