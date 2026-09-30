"""Off the Radar – daily hybrid AI discovery playlist."""

import asyncio
import json
import logging
import random
import time
from datetime import date

import httpx
import redis
from sqlalchemy.orm import Session

from app.auth import get_valid_spotify_token
from app.config import get_settings
from app.daily_drive import fetch_on_repeat_tracks
from app.database import SessionLocal
from app.gym_playlist import fetch_playlist_tracks, robust_spotify_search_with_cache
from app.models import OffTheRadarSettings, User
from app.openai_helper import generate_structured

settings = get_settings()
logger = logging.getLogger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"
TARGET_TRACK_COUNT = 30
CANDIDATE_COUNT = 40
MAX_GENERATION_ATTEMPTS = 2
HISTORY_TTL = 30 * 24 * 60 * 60
HISTORY_LIMIT = TARGET_TRACK_COUNT * 30

SONG_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "artist": {"type": "string", "minLength": 1},
    },
    "required": ["title", "artist"],
    "additionalProperties": False,
}


# ── Settings and history ──────────────────────────────

def parse_sources(serialized_sources: str | None) -> tuple[list[str], bool]:
    try:
        value = json.loads(serialized_sources or "{}")
    except (TypeError, json.JSONDecodeError):
        return [], False
    if not isinstance(value, dict):
        return [], False
    playlist_ids = value.get("playlist_ids", [])
    return [item for item in playlist_ids if isinstance(item, str)], bool(
        value.get("include_on_repeat", False)
    )


def serialize_sources(source_playlist_ids: list[str], include_on_repeat: bool) -> str:
    return json.dumps(
        {"playlist_ids": source_playlist_ids, "include_on_repeat": include_on_repeat}
    )


def history_key(user_id: int) -> str:
    return f"off_the_radar_history::{user_id}"


def get_history(user_id: int) -> list[dict]:
    """Return accepted discovery tracks from the rolling 30-day history."""
    try:
        raw = redis.Redis.from_url(settings.redis_url, decode_responses=True).get(
            history_key(user_id)
        )
        entries = json.loads(raw) if raw else []
        if not isinstance(entries, list):
            return []
        cutoff = time.time() - HISTORY_TTL
        return [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("uri"), str)
            and float(entry.get("added_at", 0)) >= cutoff
        ][:HISTORY_LIMIT]
    except Exception as error:
        logger.warning("Off the Radar: Could not load history: %s", error)
        return []


def save_history(user_id: int, tracks: list[dict]) -> None:
    """Store only tracks actually added to the discovery playlist."""
    try:
        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        now = time.time()
        cutoff = now - HISTORY_TTL
        existing = get_history(user_id)
        seen_uris = {entry["uri"] for entry in existing}
        additions = [
            {
                "uri": track["uri"],
                "title": track["title"],
                "artist": track["artist"],
                "added_at": now,
            }
            for track in tracks
            if track["uri"] not in seen_uris
        ]
        history = (additions + [entry for entry in existing if entry["added_at"] >= cutoff])[  # noqa: E203
            :HISTORY_LIMIT
        ]
        client.setex(history_key(user_id), HISTORY_TTL, json.dumps(history))
    except Exception as error:
        logger.warning("Off the Radar: Could not save history: %s", error)


# ── Hybrid curation ───────────────────────────────────

async def build_taste_profile(inspiration_songs: list[str]) -> str:
    """Use GPT-6.1 Sol to turn source songs into a compact taste profile."""
    schema = {
        "type": "object",
        "properties": {"profile": {"type": "string", "minLength": 30}},
        "required": ["profile"],
        "additionalProperties": False,
    }
    result = await generate_structured(
        model=settings.openai_curation_model,
        schema_name="off_the_radar_taste_profile",
        schema=schema,
        instructions=(
            "You are an expert music curator. Infer a concise, actionable taste profile "
            "from the supplied tracks. Identify genres, subgenres, energy, eras, languages, "
            "production details, artist neighborhoods, and discovery directions. Do not recommend songs."
        ),
        input_text="Taste-source songs:\n" + "\n".join(f"- {song}" for song in inspiration_songs),
        max_output_tokens=1200,
        reasoning_effort="low",
        retries=1,
    )
    return result["profile"]


async def generate_candidates(
    taste_profile: str,
    history: list[dict],
    source_examples: list[str],
    rejected_this_run: list[str],
) -> list[dict]:
    """Use GPT-6 Luna for high-volume candidate generation."""
    history_text = "\n".join(
        f"- {entry.get('title', '')} — {entry.get('artist', '')}" for entry in history
    ) or "- No earlier Off the Radar tracks yet"
    rejected_text = "\n".join(f"- {song}" for song in rejected_this_run[-200:]) or "- None"
    source_text = "\n".join(f"- {song}" for song in source_examples[:100])
    schema = {
        "type": "object",
        "properties": {
            "songs": {
                "type": "array",
                "items": SONG_SCHEMA,
                "minItems": CANDIDATE_COUNT,
                "maxItems": CANDIDATE_COUNT,
            }
        },
        "required": ["songs"],
        "additionalProperties": False,
    }
    prompt = f"""Taste profile:
{taste_profile}

Generate exactly {CANDIDATE_COUNT} real Spotify-available discovery candidates. They should be compelling but less obvious: explore deep cuts, overlooked artists, adjacent subgenres, and different eras without abandoning the taste profile.

The following are taste references only. Do not recommend them:
{source_text}

The following tracks have already appeared in Off the Radar during the last 30 days. Never recommend them:
{history_text}

The following candidates were rejected earlier in this generation. Never recommend them:
{rejected_text}

Rules:
- Return only real, correctly spelled songs and artists likely available on Spotify.
- Do not return duplicates.
- Avoid obvious signature hits when a more interesting fitting alternative exists.
- Return only the requested structured output."""
    result = await generate_structured(
        model=settings.openai_utility_model,
        schema_name="off_the_radar_candidates",
        schema=schema,
        instructions="You generate diverse, focused music-discovery candidates from a supplied taste profile.",
        input_text=prompt,
        max_output_tokens=8192,
        reasoning_effort="none",
        retries=1,
    )
    return result["songs"]


# ── Spotify playlist lifecycle ────────────────────────

async def _add_items(playlist_id: str, uris: list[str], headers: dict[str, str]) -> bool:
    for start in range(0, len(uris), 100):
        chunk = uris[start : start + 100]
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{SPOTIFY_API}/playlists/{playlist_id}/items",
                    headers=headers,
                    json={"uris": chunk},
                )
            if response.status_code in (200, 201):
                break
            if response.status_code == 429 and attempt < 2:
                retry_after = response.headers.get("Retry-After", "3")
                await asyncio.sleep(min(int(retry_after) if retry_after.isdigit() else 3, 30))
                continue
            logger.error("Off the Radar: Failed to add playlist items: %s", response.status_code)
            return False
    return True


async def _replace_or_create_playlist(
    existing_playlist_id: str | None,
    playlist_name: str,
    description: str,
    headers: dict[str, str],
) -> tuple[str, str]:
    if existing_playlist_id:
        async with httpx.AsyncClient(timeout=30) as client:
            check = await client.get(
                f"{SPOTIFY_API}/playlists/{existing_playlist_id}",
                headers=headers,
                params={"fields": "id"},
            )
        if check.status_code == 200:
            async with httpx.AsyncClient(timeout=30) as client:
                rename = await client.put(
                    f"{SPOTIFY_API}/playlists/{existing_playlist_id}",
                    headers=headers,
                    json={"name": playlist_name, "description": description},
                )
                clear = await client.put(
                    f"{SPOTIFY_API}/playlists/{existing_playlist_id}/items",
                    headers=headers,
                    json={"uris": []},
                )
            if rename.status_code not in (200, 201) or clear.status_code not in (200, 201):
                raise Exception("Could not update the existing Off the Radar playlist")
            return existing_playlist_id, f"https://open.spotify.com/playlist/{existing_playlist_id}"
        logger.warning("Off the Radar: Existing playlist %s unavailable (%s)", existing_playlist_id, check.status_code)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SPOTIFY_API}/me/playlists",
            headers=headers,
            json={"name": playlist_name, "description": description, "public": False},
        )
    if response.status_code not in (200, 201):
        raise Exception(f"Could not create Off the Radar playlist: {response.status_code}")
    playlist = response.json()
    await asyncio.sleep(1)
    return playlist["id"], playlist["external_urls"]["spotify"]


async def generate_off_the_radar(
    *,
    source_playlist_ids: list[str],
    include_on_repeat: bool,
    current_user: User,
    db: Session,
    existing_playlist_id: str | None = None,
) -> dict:
    """Create or refresh the user's 30-track hybrid-discovery playlist."""
    if not source_playlist_ids and not include_on_repeat:
        raise ValueError("Select at least one source playlist or enable On Repeat.")

    spotify_token = await get_valid_spotify_token(current_user, db)
    source_tracks: list[dict] = []
    for playlist_id in source_playlist_ids:
        try:
            tracks, spotify_token = await fetch_playlist_tracks(
                playlist_id, spotify_token, user=current_user, db=db
            )
            source_tracks.extend(tracks)
        except Exception as error:
            logger.warning("Off the Radar: Skipping source playlist %s: %s", playlist_id, error)

    on_repeat: list[dict] = []
    if include_on_repeat:
        on_repeat = await fetch_on_repeat_tracks(spotify_token)

    if not source_tracks and not on_repeat:
        raise ValueError("No usable songs were found in the selected sources.")

    source_uris = {track["uri"] for track in source_tracks if track.get("uri")}
    on_repeat_uris = {track["uri"] for track in on_repeat if track.get("uri")}
    prior_history = get_history(current_user.id)
    history_uris = {entry["uri"] for entry in prior_history}
    prohibited_uris = source_uris | on_repeat_uris | history_uris

    source_sample = random.sample(source_tracks, min(30, len(source_tracks))) if source_tracks else []
    repeat_sample = random.sample(on_repeat, min(20, len(on_repeat))) if on_repeat else []
    inspiration = (
        [f"[SOURCE PLAYLIST] {track['title']} — {track['artist']}" for track in source_sample]
        + [f"[ON REPEAT] {track['title']} — {track['artist']}" for track in repeat_sample]
    )
    taste_profile = await build_taste_profile(inspiration)

    accepted: list[dict] = []
    accepted_uris: set[str] = set()
    rejected: list[str] = []
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        candidates = await generate_candidates(
            taste_profile,
            prior_history,
            inspiration,
            rejected,
        )
        spotify_token = await get_valid_spotify_token(current_user, db)
        for candidate in candidates:
            if len(accepted) >= TARGET_TRACK_COUNT:
                break
            candidate_key = f"{candidate['title']} — {candidate['artist']}"
            if candidate_key.lower() in {item.lower() for item in rejected}:
                continue
            match = await robust_spotify_search_with_cache(
                candidate["title"], candidate["artist"], spotify_token
            )
            if not match or not match.get("uri"):
                rejected.append(candidate_key)
                continue
            uri = match["uri"]
            if uri in prohibited_uris or uri in accepted_uris:
                rejected.append(candidate_key)
                continue
            accepted.append({"uri": uri, "title": match["title"], "artist": match["artist"]})
            accepted_uris.add(uri)
            await asyncio.sleep(0.15)
        if len(accepted) >= TARGET_TRACK_COUNT:
            break
        logger.info(
            "Off the Radar: %s/%s valid tracks after candidate pass %s",
            len(accepted),
            TARGET_TRACK_COUNT,
            attempt + 1,
        )

    if len(accepted) < TARGET_TRACK_COUNT:
        raise Exception(
            f"Could only find {len(accepted)} fresh, verified tracks. Please try again."
        )

    playlist_name = f"Off the Radar – {date.today().strftime('%d.%m.%Y')}"
    description = "30 fresh paths beyond your current rotation, curated daily by VibeSwipe."
    spotify_token = await get_valid_spotify_token(current_user, db)
    headers = {"Authorization": f"Bearer {spotify_token}"}
    playlist_id, playlist_url = await _replace_or_create_playlist(
        existing_playlist_id, playlist_name, description, headers
    )
    add_success = await _add_items(playlist_id, [track["uri"] for track in accepted], headers)
    if not add_success:
        raise Exception("Spotify could not add all Off the Radar tracks.")

    save_history(current_user.id, accepted)
    return {
        "playlist_id": playlist_id,
        "playlist_url": playlist_url,
        "playlist_name": playlist_name,
        "total_tracks": len(accepted),
        "inspiration_count": len(inspiration),
        "new_discoveries_count": len(accepted),
        "taste_profile": taste_profile,
    }


async def auto_refresh_off_the_radar_playlists() -> None:
    """Refresh all opted-in Off the Radar playlists at 05:00."""
    logger.info("Off the Radar Auto-Refresh: Starting")
    db: Session = SessionLocal()
    try:
        settings_rows = (
            db.query(OffTheRadarSettings)
            .filter(OffTheRadarSettings.auto_refresh == True)  # noqa: E712
            .all()
        )
        for radar_settings in settings_rows:
            try:
                user = db.query(User).filter(User.id == radar_settings.user_id).first()
                if not user:
                    continue
                source_ids, include_on_repeat = parse_sources(radar_settings.source_playlist_ids)
                result = await generate_off_the_radar(
                    source_playlist_ids=source_ids,
                    include_on_repeat=include_on_repeat,
                    current_user=user,
                    db=db,
                    existing_playlist_id=radar_settings.last_spotify_playlist_id,
                )
                radar_settings.last_spotify_playlist_id = result["playlist_id"]
                db.commit()
                await asyncio.sleep(5)
            except Exception as error:
                db.rollback()
                logger.error(
                    "Off the Radar Auto-Refresh: Failed for user %s: %s",
                    radar_settings.user_id,
                    error,
                    exc_info=True,
                )
    finally:
        db.close()
    logger.info("Off the Radar Auto-Refresh: Done")
