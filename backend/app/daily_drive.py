"""
Daily Drive generator – custom "Daily Drive" playlist.

Flow:
1. Fetch user's top tracks (short_term = "On Repeat") from Spotify
2. Fetch user's saved shows (podcasts)
3. User picks which shows to include
4. Gemini receives all On-Repeat song titles and returns:
   - 20 songs from the On-Repeat list (shuffled selection)
   - 20 NEW songs that match the same vibe but aren't in On-Repeat
5. Fetch random recent episodes from the selected shows
6. Interleave: 4 songs → 1 podcast episode → 4 songs → 1 podcast episode …
7. Create a Spotify playlist with the result
"""

import json
import random
import asyncio
import logging
from datetime import date

import httpx
import redis
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.1-pro-preview:generateContent?key={settings.gemini_api_key}"
)

# Redis Client initialisieren
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

# History TTL: 5 days – keeps track of recently used Daily Drive songs
HISTORY_TTL = 5 * 24 * 60 * 60  # 5 days in seconds


# Hilfsfunktion: Key für Song generieren
def song_cache_key(title: str, artist: str) -> str:
    return f"song_uri::{title.lower().strip()}|||{artist.lower().strip()}"


def daily_drive_history_key(user_id: int) -> str:
    return f"daily_drive_history::{user_id}"


def save_daily_drive_history(user_id: int, songs: list[dict]) -> None:
    """Save generated Daily Drive songs to Redis so they can be avoided next time.
    Songs expire after 5 days automatically via TTL."""
    key = daily_drive_history_key(user_id)
    try:
        entries = [f"{s['title'].lower().strip()} - {s['artist'].lower().strip()}" for s in songs]
        existing = set(redis_client.lrange(key, 0, -1))
        for entry in entries:
            if entry not in existing:
                redis_client.rpush(key, entry)
        # Cap history at 200 entries to avoid bloat, keep newest
        current_len = redis_client.llen(key)
        if current_len > 200:
            redis_client.ltrim(key, current_len - 200, -1)
        redis_client.expire(key, HISTORY_TTL)
        logger.info(f"Daily Drive History: Saved {len(entries)} songs for user {user_id} (total: {redis_client.llen(key)})")
    except Exception as e:
        logger.warning(f"Daily Drive History: Could not save history: {e}")


def get_daily_drive_history(user_id: int) -> list[str]:
    """Get recently used Daily Drive songs from Redis (last 5 days)."""
    key = daily_drive_history_key(user_id)
    try:
        history = redis_client.lrange(key, 0, -1)
        logger.info(f"Daily Drive History: Found {len(history)} recent songs for user {user_id}")
        return history
    except Exception as e:
        logger.warning(f"Daily Drive History: Could not read history: {e}")
        return []

async def fetch_on_repeat_tracks(spotify_token: str) -> list[dict]:
    """Fetch user's top tracks from short_term AND medium_term for a bigger pool.
    short_term ≈ On Repeat (last ~4 weeks), medium_term ≈ last ~6 months.
    Merges both, deduplicates by URI, short_term first (higher priority)."""
    all_tracks = []
    seen_uris: set[str] = set()

    async with httpx.AsyncClient() as client:
        # Fetch short_term first (most recent favorites)
        for time_range in ["short_term", "medium_term"]:
            resp = await client.get(
                f"{SPOTIFY_API}/me/top/tracks",
                params={"time_range": time_range, "limit": 50},
                headers={"Authorization": f"Bearer {spotify_token}"},
            )
            if resp.status_code != 200:
                if time_range == "short_term":
                    logger.error(f"Failed to fetch top tracks ({time_range}): {resp.status_code} {resp.text[:300]}")
                    raise Exception(f"Could not fetch On-Repeat tracks: {resp.status_code}")
                else:
                    logger.warning(f"Failed to fetch medium_term tracks: {resp.status_code} (non-fatal)")
                    continue

            data = resp.json()
            for item in data.get("items", []):
                uri = item["uri"]
                if uri not in seen_uris:
                    seen_uris.add(uri)
                    all_tracks.append({
                        "title": item["name"],
                        "artist": ", ".join(a["name"] for a in item.get("artists", [])),
                        "uri": uri,
                        "id": item["id"],
                    })

    logger.info(f"Daily Drive: Fetched {len(all_tracks)} unique top tracks (short + medium term)")
    return all_tracks


async def fetch_saved_shows(spotify_token: str) -> list[dict]:
    """Fetch user's saved podcast shows."""
    shows = []
    url = f"{SPOTIFY_API}/me/shows"
    params: dict = {"limit": 50}

    async with httpx.AsyncClient() as client:
        while url:
            resp = await client.get(
                url, params=params,
                headers={"Authorization": f"Bearer {spotify_token}"},
            )
            if resp.status_code != 200:
                logger.error(f"Failed to fetch saved shows: {resp.status_code} {resp.text[:300]}")
                raise Exception(f"Could not fetch saved shows: {resp.status_code}")

            data = resp.json()
            for item in data.get("items", []):
                show = item.get("show", {})
                images = show.get("images", [])
                shows.append({
                    "id": show["id"],
                    "name": show.get("name", ""),
                    "publisher": show.get("publisher", ""),
                    "image": images[0]["url"] if images else None,
                    "total_episodes": show.get("total_episodes", 0),
                })
            url = data.get("next")
            params = {}

    return shows


async def fetch_show_episodes(show_id: str, spotify_token: str, limit: int = 50) -> list[dict]:
    """
    Fetch the most recent UNPLAYED episodes of a show.

    Strategy: fetch episodes sorted newest-first (Spotify default).
    Skip any episode that has been fully played (resume_point.fully_played).
    Return the first `needed` unplayed episodes.
    If all fetched episodes are played, paginate to older ones.
    Requires scope 'user-read-playback-position' for resume_point data.
    """
    headers = {"Authorization": f"Bearer {spotify_token}"}
    episodes: list[dict] = []
    offset = 0
    max_pages = 5  # Safety limit – don't paginate forever

    async with httpx.AsyncClient() as client:
        for _ in range(max_pages):
            resp = await client.get(
                f"{SPOTIFY_API}/shows/{show_id}/episodes",
                params={"limit": limit, "offset": offset, "market": "DE"},
                headers=headers,
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch episodes for show {show_id}: {resp.status_code}")
                break

            data = resp.json()
            items = data.get("items", [])
            if not items:
                break

            for ep in items:
                # Check if episode was fully played
                resume_point = ep.get("resume_point", {})
                fully_played = resume_point.get("fully_played", False) if resume_point else False

                episodes.append({
                    "name": ep.get("name", ""),
                    "uri": ep.get("uri", ""),
                    "id": ep.get("id", ""),
                    "duration_ms": ep.get("duration_ms", 0),
                    "release_date": ep.get("release_date", ""),
                    "fully_played": fully_played,
                    "show_id": show_id,
                })

            # If there are no more pages, stop
            if not data.get("next"):
                break

            offset += limit

    return episodes


async def ask_gemini_daily_drive(on_repeat_songs: list[dict], recent_history: list[str] | None = None) -> dict:
    """Ask Gemini to curate the Daily Drive song selection.
    If recent_history is provided, Gemini will avoid those songs for better variation."""
    song_list = "\n".join(
        f"- {s['title']} – {s['artist']}" for s in on_repeat_songs
    )

    num_from_repeat = min(20, len(on_repeat_songs))
    num_new = 20

    # Build avoidance block from history
    avoid_block = ""
    if recent_history:
        avoid_list = "\n".join(f"- {s}" for s in recent_history[:150])  # cap to avoid prompt overflow
        avoid_block = f"""

CRITICAL – AVOID REPEATS: The following songs were already used in recent Daily Drive playlists (last few days).
You MUST NOT include ANY of these songs in your "new_discoveries" list. Pick COMPLETELY DIFFERENT songs instead.
Also try to pick DIFFERENT songs from the on-repeat list than what was used before (choose ones not in this avoid list):
{avoid_list}
"""

    # Generate a random seed to encourage different outputs each time
    seed_hint = random.randint(1000, 9999)

    prompt = f"""You are a music curation expert building a "Daily Drive" playlist.
Today's session seed: {seed_hint} (use this to vary your picks – be creative and surprising!)

I will give you a list of songs that the user currently has on repeat (their favorite songs right now).

Your task:
1. Pick exactly {num_from_repeat} songs FROM the provided list. Choose a DIFFERENT selection each time – don't always pick the most popular or obvious ones. Rotate through the full list. Use the EXACT titles and artists as given.
2. Recommend exactly {num_new} NEW songs that are NOT in the provided list but perfectly match the style, mood, genre, and energy of these songs. These should be songs the user would likely enjoy but hasn't discovered yet. Be CREATIVE and DIVERSE – explore different sub-genres, eras, and lesser-known tracks.
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
- "from_repeat" must contain exactly {num_from_repeat} songs that are IN the provided list (use the exact titles/artists given)
- "new_discoveries" must contain exactly {num_new} songs NOT in the provided list
- MAXIMIZE VARIETY: Mix genres, eras (80s, 90s, 2000s, 2010s, 2020s), and energy levels
- Include some deep cuts and lesser-known tracks, not just the most obvious picks
- Create a listening experience that feels FRESH each time
- No duplicates within the list
- Only output valid JSON, no markdown, no explanation"""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"text": f"Here are the user's On-Repeat songs:\n{song_list}"},
            ]
        }],
        "generationConfig": {
            "temperature": 1.8,
            "maxOutputTokens": 8192,
            "topP": 0.95,
            "topK": 64,
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(GEMINI_URL, json=payload)

    if resp.status_code != 200:
        logger.error(f"Gemini API error: {resp.status_code} – {resp.text[:500]}")
        raise Exception(f"Gemini API error: {resp.status_code} – {resp.text[:200]}")

    data = resp.json()
    
    # Safely extract text from Gemini response
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected Gemini response structure: {json.dumps(data)[:500]}")
        raise Exception(f"Unexpected Gemini response: {e}")

    # Strip markdown code fences if present
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


async def robust_spotify_search(query: str, spotify_token: str, max_retries: int = 3) -> dict | None:
    """Spotify search with retry-after handling and 429 logging."""
    headers = {"Authorization": f"Bearer {spotify_token}"}
    params = {"q": query, "type": "track", "limit": 1}
    for attempt in range(max_retries):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{SPOTIFY_API}/search",
                params=params,
                headers=headers,
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
                "id": track["id"],
            }
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "3")
            try:
                wait = min(int(retry_after), 30)
            except ValueError:
                wait = 3
            logger.warning(f"Spotify search 429 for '{query}', Retry-After={retry_after}s, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(wait)
            continue
        logger.warning(f"Spotify search failed for '{query}': {resp.status_code}")
        return None
    logger.error(f"Spotify search gave up after {max_retries} retries for '{query}'")
    return None


async def robust_add_items_to_playlist(client, playlist_id, chunk, auth_headers, max_retries=3):
    """Add items to playlist with Retry-After handling and logging."""
    for attempt in range(max_retries):
        add_resp = await client.post(
            f"{SPOTIFY_API}/playlists/{playlist_id}/items",
            headers=auth_headers,
            json={"uris": chunk},
        )
        if add_resp.status_code in (200, 201):
            return True
        if add_resp.status_code == 429:
            retry_after = add_resp.headers.get("Retry-After", "3")
            try:
                wait = min(int(retry_after), 30)
            except ValueError:
                wait = 3
            logger.warning(f"Playlist add 429, Retry-After={retry_after}s, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(wait)
            continue
        logger.error(f"Failed to add items to playlist: {add_resp.status_code} {add_resp.text[:300]}")
        return False
    logger.error(f"Failed to add items after {max_retries} retries.")
    return False


# Robust Spotify-Search mit Redis-Cache
async def robust_spotify_search_with_cache(title: str, artist: str, spotify_token: str, max_retries: int = 3) -> dict | None:
    key = song_cache_key(title, artist)
    uri = redis_client.get(key)
    if uri:
        # Prüfe, ob URI gültig ist (optional: API-Check, hier nur Format)
        if uri.startswith("spotify:track:"):
            logger.info(f"Cache hit for '{title} {artist}': {uri}")
            return {"title": title, "artist": artist, "uri": uri}
        else:
            logger.warning(f"Cache invalid URI for '{title} {artist}': {uri}")
    # Fallback: Suche via API
    search_result = await robust_spotify_search(f"{title} {artist}", spotify_token, max_retries)
    if search_result and search_result.get("uri"):
        redis_client.set(key, search_result["uri"])
        logger.info(f"Cache set for '{title} {artist}': {search_result['uri']}")
        return search_result
    logger.warning(f"No URI found for '{title} {artist}'")
    return None


async def generate_daily_drive(
    spotify_token: str,
    spotify_user_id: str,
    selected_show_ids: list[str],
    user_id: int | None = None,
) -> dict:
    """
    Full Daily Drive generation pipeline.
    Returns info about the created playlist.
    """
    # 1. Fetch On-Repeat tracks
    logger.info("Daily Drive: Fetching On-Repeat tracks...")
    on_repeat = await fetch_on_repeat_tracks(spotify_token)
    logger.info(f"Daily Drive: Got {len(on_repeat)} On-Repeat tracks")
    if len(on_repeat) < 5:
        raise Exception(
            "You need at least 5 songs in your Top Tracks (On Repeat). "
            "Listen to more music and try again later!"
        )

    # 2. Fetch podcast episodes EARLY (before Gemini) so the rate limit
    #    has time to recover while Gemini processes (~5-10s)
    unplayed_episodes: list[dict] = []
    played_episodes: list[dict] = []
    if selected_show_ids:
        logger.info(f"Daily Drive: Fetching episodes for {len(selected_show_ids)} shows (sequentially)...")
        for show_id in selected_show_ids:
            eps = await fetch_show_episodes(show_id, spotify_token, limit=20)
            for ep in eps:
                if ep["fully_played"]:
                    played_episodes.append(ep)
                else:
                    unplayed_episodes.append(ep)
            # Small pause between shows to avoid rate limiting
            if len(selected_show_ids) > 1:
                await asyncio.sleep(0.5)
        logger.info(f"Daily Drive: Found {len(unplayed_episodes)} unplayed + {len(played_episodes)} played episodes")

    # 3. Load recent song history & ask Gemini to curate (NO Spotify API calls → rate limit recovers here!)
    recent_history = get_daily_drive_history(user_id) if user_id else []
    logger.info(f"Daily Drive: {len(recent_history)} songs in 5-day history to avoid")
    logger.info("Daily Drive: Asking Gemini to curate songs...")
    gemini_result = await ask_gemini_daily_drive(on_repeat, recent_history if recent_history else None)
    logger.info(
        f"Daily Drive: Gemini returned {len(gemini_result.get('from_repeat', []))} from_repeat, "
        f"{len(gemini_result.get('new_discoveries', []))} new_discoveries"
    )

    # 4. Map "from_repeat" songs back to their Spotify URIs (no API calls needed)
    on_repeat_map = {}
    for t in on_repeat:
        key = f"{t['title'].lower().strip()}|||{t['artist'].lower().strip()}"
        on_repeat_map[key] = t
        # Also store by title only for fuzzy matching
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

    logger.info(f"Daily Drive: Matched {len(from_repeat_uris)} from_repeat directly, {len(unmatched_from_repeat)} need search")

    # 5. Search unmatched from_repeat + all new discoveries on Spotify
    #    SEQUENTIELL: Jeder Song wird einzeln gesucht, mit kurzem Delay
    all_to_search: list[dict] = []
    for song in unmatched_from_repeat:
        all_to_search.append({"song": song, "type": "from_repeat"})
    for song in gemini_result.get("new_discoveries", []):
        all_to_search.append({"song": song, "type": "new_discovery"})

    logger.info(f"Daily Drive: Searching {len(all_to_search)} songs on Spotify (sequentiell)...")

    new_discovery_uris: list[str] = []
    results: list[tuple[str, str | None]] = []
    for item in all_to_search:
        song = item["song"]
        search_result = await robust_spotify_search_with_cache(song['title'], song['artist'], spotify_token)
        uri = search_result["uri"] if search_result else None
        results.append((item["type"], uri))
        await asyncio.sleep(1.2)  # Kurze Pause nach jedem Call

    for typ, uri in results:
        if uri is None:
            continue
        if typ == "from_repeat":
            from_repeat_uris.append(uri)
        else:
            new_discovery_uris.append(uri)

    logger.info(f"Daily Drive: Final counts – {len(from_repeat_uris)} from_repeat, {len(new_discovery_uris)} new discoveries")

    # 5b. Deduplicate: remove URIs that appear in both lists and remove any duplicates
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
    logger.info(f"Daily Drive: After dedup – {len(from_repeat_uris)} from_repeat, {len(new_discovery_uris)} new discoveries")

    # 5c. Save all songs to history for future avoidance
    if user_id:
        all_songs_for_history = []
        for song in gemini_result.get("from_repeat", []):
            all_songs_for_history.append(song)
        for song in gemini_result.get("new_discoveries", []):
            all_songs_for_history.append(song)
        save_daily_drive_history(user_id, all_songs_for_history)

    # 6. Combine: shuffle both sets for variety
    random.shuffle(from_repeat_uris)
    random.shuffle(new_discovery_uris)

    # Interleave: alternate from_repeat and new_discoveries
    all_song_uris: list[str] = []
    repeat_iter = iter(from_repeat_uris)
    new_iter = iter(new_discovery_uris)
    use_repeat = True
    while True:
        src = repeat_iter if use_repeat else new_iter
        uri = next(src, None)
        if uri is None:
            # Switch to the other source for remaining songs
            other = new_iter if use_repeat else repeat_iter
            for remaining in other:
                all_song_uris.append(remaining)
            break
        all_song_uris.append(uri)
        use_repeat = not use_repeat

    # 7. Pick podcast episodes from the data fetched in step 2
    episode_uris: list[str] = []
    if selected_show_ids:
        # Sort both lists by release_date descending (newest first)
        unplayed_episodes.sort(key=lambda e: e.get("release_date", ""), reverse=True)
        played_episodes.sort(key=lambda e: e.get("release_date", ""), reverse=True)

        # We need roughly len(all_song_uris) / 4 episodes
        needed = max(1, len(all_song_uris) // 4)

        # Prefer unplayed episodes (newest first), fall back to played if not enough
        chosen_episodes = unplayed_episodes[:needed]
        if len(chosen_episodes) < needed:
            remaining = needed - len(chosen_episodes)
            chosen_episodes.extend(played_episodes[:remaining])

        logger.info(
            f"Daily Drive: Picked {len(chosen_episodes)} episodes "
            f"({min(needed, len(unplayed_episodes))} unplayed, "
            f"{max(0, len(chosen_episodes) - len(unplayed_episodes))} played fallback)"
        )

        episode_uris = [ep["uri"] for ep in chosen_episodes]

    # 8. Interleave: 4 songs → 1 episode → 4 songs → 1 episode …
    final_uris: list[str] = []
    song_idx = 0
    ep_idx = 0

    while song_idx < len(all_song_uris):
        # Add up to 4 songs
        chunk = all_song_uris[song_idx: song_idx + 4]
        final_uris.extend(chunk)
        song_idx += 4

        # Add 1 episode if available
        if ep_idx < len(episode_uris):
            final_uris.append(episode_uris[ep_idx])
            ep_idx += 1

    # 9. Create the Spotify playlist
    logger.info(f"Daily Drive: Creating playlist with {len(final_uris)} items...")
    today = date.today().strftime("%d.%m.%Y")
    playlist_name = f"Daily Drive – {today}"
    playlist_desc = (
        f"Your personal Daily Drive by VibeSwipe 🚗 "
        f"{len(from_repeat_uris)} On-Repeat Songs, "
        f"{len(new_discovery_uris)} new discoveries"
        f"{f', {len(episode_uris)} podcast episodes' if episode_uris else ''}"
    )

    auth_headers = {"Authorization": f"Bearer {spotify_token}"}

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
            raise Exception(f"Could not create playlist: {create_resp.text}")

        playlist = create_resp.json()
        playlist_id = playlist["id"]

        # Add items in chunks of 100
        for i in range(0, len(final_uris), 100):
            chunk = final_uris[i: i + 100]
            success = await robust_add_items_to_playlist(client, playlist_id, chunk, auth_headers)
            if not success:
                logger.error(f"Failed to add chunk {i}-{i+len(chunk)} to playlist after retries.")

    return {
        "playlist_url": playlist["external_urls"]["spotify"],
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "total_tracks": len(final_uris),
        "on_repeat_count": len(from_repeat_uris),
        "new_discoveries_count": len(new_discovery_uris),
        "episodes_count": len(episode_uris),
    }
