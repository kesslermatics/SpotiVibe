from urllib.parse import urlencode
import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.config import get_settings
from app.database import get_db
from app.models import User
from app.schemas import SpotifyCallback, Token, UserResponse, MessageResponse, DiscoverRequest, DiscoverResponse, CreatePlaylistRequest, CreatePlaylistResponse, SaveTracksRequest, SaveTracksResponse, DailyDriveRequest, DailyDriveResponse, GymPlaylistGenerateRequest, GymPlaylistGenerateResponse, GymPlaylistSettingsResponse, GymPlaylistAutoRefreshRequest, SwipeDeckResponse, RoastResponse
from app.auth import create_access_token, get_current_user, get_valid_spotify_token, refresh_spotify_token
from app.discover import discover_songs
from app.daily_drive import fetch_saved_shows, generate_daily_drive, fetch_on_repeat_tracks
from app.gym_playlist import generate_gym_playlist
from app.roast import generate_vibe_roast
from app.models import GymPlaylistSettings
import json

router = APIRouter()
settings = get_settings()

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_ME_URL = "https://api.spotify.com/v1/me"
SPOTIFY_SCOPES = "user-read-email user-read-private user-top-read user-library-read user-library-modify playlist-read-private playlist-read-collaborative playlist-modify-public playlist-modify-private user-read-playback-position"


def _normalize_uri(uri: str) -> str:
    """Spotify requires 127.0.0.1 instead of localhost for local dev."""
    return uri.replace("://localhost:", "://127.0.0.1:")


# ── Spotify OAuth: Get login URL ─────────────────────
@router.get("/auth/login")
def spotify_login(redirect_uri: str | None = None):
    """Returns the Spotify authorization URL the frontend should redirect to."""
    allowed = [_normalize_uri(u) for u in settings.spotify_redirect_uris]

    # Default to first allowed URI
    uri = allowed[0]
    if redirect_uri:
        normalized = _normalize_uri(redirect_uri)
        if normalized in allowed:
            uri = normalized

    params = {
        "client_id": settings.spotify_client_id,
        "response_type": "code",
        "redirect_uri": uri,
        "scope": SPOTIFY_SCOPES,
        "show_dialog": "true",
    }
    return {"url": f"{SPOTIFY_AUTH_URL}?{urlencode(params)}", "redirect_uri": uri}


# ── Spotify OAuth: Exchange code for tokens ───────────
@router.post("/auth/callback", response_model=Token)
async def spotify_callback(payload: SpotifyCallback, db: Session = Depends(get_db)):
    """Exchange the Spotify auth code for tokens, create/update user, return JWT."""

    # Validate redirect_uri (normalize localhost ↔ 127.0.0.1)
    allowed = [_normalize_uri(u) for u in settings.spotify_redirect_uris]
    resolved_uri = _normalize_uri(payload.redirect_uri)
    if resolved_uri not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid redirect_uri",
        )

    # 1. Exchange code for Spotify access & refresh tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            SPOTIFY_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": payload.code,
                "redirect_uri": resolved_uri,
                "client_id": settings.spotify_client_id,
                "client_secret": settings.spotify_client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if token_resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Spotify token exchange failed: {token_resp.text}",
        )

    token_data = token_resp.json()
    spotify_access_token = token_data["access_token"]
    spotify_refresh_token = token_data.get("refresh_token")

    # 2. Fetch user profile from Spotify
    async with httpx.AsyncClient() as client:
        me_resp = await client.get(
            SPOTIFY_ME_URL,
            headers={"Authorization": f"Bearer {spotify_access_token}"},
        )

    if me_resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to fetch Spotify user profile",
        )

    me = me_resp.json()
    spotify_id = me["id"]
    email = me.get("email")
    display_name = me.get("display_name")

    # 3. Create or update user in DB
    user = db.query(User).filter(User.spotify_id == spotify_id).first()
    if user:
        user.email = email
        user.display_name = display_name
        user.spotify_access_token = spotify_access_token
        user.spotify_refresh_token = spotify_refresh_token or user.spotify_refresh_token
    else:
        user = User(
            spotify_id=spotify_id,
            email=email,
            display_name=display_name,
            spotify_access_token=spotify_access_token,
            spotify_refresh_token=spotify_refresh_token,
        )
        db.add(user)

    db.commit()
    db.refresh(user)

    # 4. Issue our own JWT
    access_token = create_access_token(data={"sub": user.spotify_id})
    return {"access_token": access_token, "token_type": "bearer"}


# ── Get current user info ────────────────────────────
@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# ── Discover songs via Gemini + Spotify ──────────────
@router.post("/discover", response_model=DiscoverResponse)
async def discover(
    payload: DiscoverRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    spotify_token = await get_valid_spotify_token(current_user, db)

    try:
        # Fetch user's top tracks if "include my taste" is enabled
        on_repeat_songs = None
        if payload.include_my_taste:
            try:
                on_repeat_songs = await fetch_on_repeat_tracks(spotify_token)
            except Exception as e:
                logger.warning(f"Could not fetch on-repeat tracks: {e}")

        result = await discover_songs(
            payload.prompt,
            spotify_token,
            context_songs=payload.context_songs or None,
            on_repeat_songs=on_repeat_songs,
            save_to_playlist=payload.save_to_playlist,
        )

        # Create Spotify playlist if "save to playlist" is enabled
        if payload.save_to_playlist:
            track_uris = [s["spotify_uri"] for s in result["songs"] if s.get("spotify_uri")]
            if track_uris:
                playlist_name = result.get("playlist_name") or "Discover Mix"
                playlist_desc = result.get("playlist_description") or result.get("mood_summary", "")

                # Refresh token (may have gone stale during Gemini + search pipeline)
                spotify_token = await get_valid_spotify_token(current_user, db)

                # Create playlist via /me/playlists (works in dev mode)
                async with httpx.AsyncClient() as client:
                    create_resp = await client.post(
                        f"{SPOTIFY_API_BASE}/me/playlists",
                        headers={"Authorization": f"Bearer {spotify_token}"},
                        json={
                            "name": playlist_name,
                            "description": playlist_desc,
                            "public": False,
                        },
                    )
                if create_resp.status_code not in (200, 201):
                    logger.error(f"Create playlist failed: {create_resp.status_code} {create_resp.text[:300]}")
                    raise Exception(f"Could not create playlist: {create_resp.status_code}")

                playlist_data = create_resp.json()
                playlist_id = playlist_data["id"]
                playlist_url = playlist_data["external_urls"]["spotify"]

                # Add tracks in chunks of 100
                for i in range(0, len(track_uris), 100):
                    chunk = track_uris[i:i + 100]
                    async with httpx.AsyncClient() as client:
                        add_resp = await client.post(
                            f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks",
                            headers={
                                "Authorization": f"Bearer {spotify_token}",
                                "Content-Type": "application/json",
                            },
                            json={"uris": chunk},
                        )
                    if add_resp.status_code not in (200, 201):
                        logger.error(f"Add tracks failed: {add_resp.status_code} {add_resp.text[:300]}")

                result["playlist_url"] = playlist_url
                result["playlist_id"] = playlist_id
                result["playlist_name"] = playlist_name

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Discovery failed: {str(e)}",
        )


# ── Fetch tracks from a Spotify playlist ─────────────
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


@router.get("/my-playlists")
async def get_my_playlists(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all playlists belonging to the current user."""
    spotify_token = await get_valid_spotify_token(current_user, db)

    playlists: list[dict] = []
    url = f"{SPOTIFY_API_BASE}/me/playlists"
    params: dict = {"limit": 50}

    async with httpx.AsyncClient() as client:
        while url:
            resp = await client.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {spotify_token}"},
            )
            if resp.status_code != 200:
                logger.error(f"my-playlists failed: status={resp.status_code}, body={resp.text[:300]}")
                raise HTTPException(status_code=resp.status_code, detail=f"Could not load playlists: {resp.text[:200]}")

            data = resp.json()
            for item in data.get("items", []):
                if not item:
                    continue
                images = item.get("images") or []
                tracks_obj = item.get("tracks") or {}
                playlists.append({
                    "id": item["id"],
                    "name": item.get("name", ""),
                    "image": images[0]["url"] if images else None,
                    "total_tracks": tracks_obj.get("total", 0) if isinstance(tracks_obj, dict) else 0,
                    "owner": (item.get("owner") or {}).get("display_name", ""),
                })

            url = data.get("next")
            params = {}

    return {"playlists": playlists}


@router.get("/playlist-tracks")
async def get_playlist_tracks(
    playlist_id: str | None = None,
    playlist_url: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Extract track names from a Spotify playlist URL or ID."""
    spotify_token = await get_valid_spotify_token(current_user, db)

    # Resolve the playlist ID from whichever param was provided
    resolved_id = playlist_id
    if not resolved_id and playlist_url:
        resolved_id = playlist_url.strip()
        if "spotify.com/playlist/" in resolved_id:
            resolved_id = resolved_id.split("playlist/")[1].split("?")[0].split("/")[0]
        elif "spotify:playlist:" in resolved_id:
            resolved_id = resolved_id.split(":")[-1]

    if not resolved_id:
        raise HTTPException(status_code=400, detail="playlist_id or playlist_url is required.")

    songs: list[str] = []
    url = f"{SPOTIFY_API_BASE}/playlists/{resolved_id}/items"
    params: dict = {"limit": 50}

    async with httpx.AsyncClient() as client:
        while url:
            resp = await client.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {spotify_token}"},
            )
            logger.info(f"playlist-tracks first attempt: status={resp.status_code}, playlist={resolved_id}")
            # If 401/403, try refreshing the token once
            if resp.status_code in (401, 403):
                logger.warning(f"playlist-tracks got {resp.status_code}, response: {resp.text[:300]}")
                spotify_token = await refresh_spotify_token(current_user, db)
                resp = await client.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {spotify_token}"},
                )
                logger.info(f"playlist-tracks after refresh: status={resp.status_code}")
            if resp.status_code != 200:
                error_body = resp.text[:500]
                logger.error(f"playlist-tracks failed: status={resp.status_code}, body={error_body}")
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Spotify API error ({resp.status_code}): {error_body}",
                )

            data = resp.json()
            for item in data.get("items", []):
                track = item.get("track") or item.get("item")
                if track and track.get("name"):
                    artist = ", ".join(a["name"] for a in track.get("artists", []))
                    songs.append(f"{track['name']} - {artist}")

            url = data.get("next")
            params = {}  # next URL already includes params

    return {"songs": songs, "total": len(songs)}


@router.post("/create-playlist", response_model=CreatePlaylistResponse)
async def create_playlist(
    payload: CreatePlaylistRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    spotify_token = await get_valid_spotify_token(current_user, db)
    headers = {"Authorization": f"Bearer {spotify_token}"}

    try:
        async with httpx.AsyncClient() as client:
            # 1. Create an empty playlist on the user's account
            create_resp = await client.post(
                f"{SPOTIFY_API_BASE}/me/playlists",
                headers=headers,
                json={
                    "name": payload.name,
                    "description": payload.description,
                    "public": False,
                },
            )

        if create_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create playlist: {create_resp.text}",
            )

        playlist = create_resp.json()
        playlist_id = playlist["id"]

        # 2. Add tracks in chunks of 100 (Spotify limit)
        for i in range(0, len(payload.track_uris), 100):
            chunk = payload.track_uris[i : i + 100]
            async with httpx.AsyncClient() as client:
                add_resp = await client.post(
                    f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks",
                    headers=headers,
                    json={"uris": chunk},
                )
            if add_resp.status_code not in (200, 201):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to add tracks: {add_resp.text}",
                )

        return {
            "playlist_url": playlist["external_urls"]["spotify"],
            "playlist_id": playlist_id,
            "name": payload.name,
            "total_tracks": len(payload.track_uris),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Playlist creation failed: {str(e)}",
        )


# ── Save tracks to Liked Songs ───────────────────────
@router.post("/save-tracks", response_model=SaveTracksResponse)
async def save_tracks(
    payload: SaveTracksRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    spotify_token = await get_valid_spotify_token(current_user, db)
    headers = {"Authorization": f"Bearer {spotify_token}"}
    track_uris = [f"spotify:track:{tid}" for tid in payload.track_ids]

    try:
        # 1. Try saving directly to Liked Songs first
        saved_directly = True
        for i in range(0, len(payload.track_ids), 50):
            chunk = payload.track_ids[i : i + 50]
            async with httpx.AsyncClient() as client:
                save_resp = await client.put(
                    f"{SPOTIFY_API_BASE}/me/tracks",
                    headers=headers,
                    json={"ids": chunk},
                )
            if save_resp.status_code not in (200, 201):
                saved_directly = False
                break

        if saved_directly:
            # Check how many were already saved (best-effort)
            already_saved = 0
            return {
                "saved": len(payload.track_ids),
                "already_saved": already_saved,
            }

        # 2. Fallback: Create a playlist instead
        async with httpx.AsyncClient() as client:
            create_resp = await client.post(
                f"{SPOTIFY_API_BASE}/me/playlists",
                headers=headers,
                json={
                    "name": f"SpotiVibe Discover – {len(track_uris)} Songs",
                    "description": "Created with SpotiVibe AI Discover",
                    "public": False,
                },
            )

        if create_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not save to Liked Songs or create playlist: {create_resp.text}",
            )

        playlist = create_resp.json()
        playlist_id = playlist["id"]

        for i in range(0, len(track_uris), 100):
            chunk = track_uris[i : i + 100]
            async with httpx.AsyncClient() as client:
                add_resp = await client.post(
                    f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks",
                    headers=headers,
                    json={"uris": chunk},
                )

        return {
            "saved": len(payload.track_ids),
            "already_saved": 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Saving tracks failed: {str(e)}",
        )


# ── Daily Drive: Fetch saved shows (podcasts) ────────
@router.get("/daily-drive/shows")
async def get_saved_shows(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the user's saved podcast shows for Daily Drive selection."""
    spotify_token = await get_valid_spotify_token(current_user, db)
    try:
        shows = await fetch_saved_shows(spotify_token)
        return {"shows": shows}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch shows: {str(e)}",
        )


# ── Daily Drive: Generate playlist ───────────────────
@router.post("/daily-drive/generate", response_model=DailyDriveResponse)
async def generate_daily_drive_playlist(
    payload: DailyDriveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a custom Daily Drive playlist."""
    spotify_token = await get_valid_spotify_token(current_user, db)
    try:
        result = await generate_daily_drive(
            spotify_token=spotify_token,
            spotify_user_id=current_user.spotify_id,
            selected_show_ids=payload.selected_show_ids,
        )
        return result
    except Exception as e:
        logger.error(f"Daily Drive generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Daily Drive creation failed: {str(e)}",
        )


# ── Gym Playlist: Generate ───────────────────────────
@router.post("/gym-playlist/generate", response_model=GymPlaylistGenerateResponse)
async def gym_playlist_generate(
    payload: GymPlaylistGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = await generate_gym_playlist(
            source_playlist_ids=payload.source_playlist_ids,
            current_user=current_user,
            db=db,
        )
        return result
    except Exception as e:
        logger.error(f"Gym playlist generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gym Playlist creation failed: {str(e)}",
        )


# ── Gym Playlist: Get settings ───────────────────────
@router.get("/gym-playlist/settings", response_model=GymPlaylistSettingsResponse)
def gym_playlist_get_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        gym_settings = (
            db.query(GymPlaylistSettings)
            .filter(GymPlaylistSettings.user_id == current_user.id)
            .first()
        )
    except Exception as e:
        logger.warning(f"Could not query GymPlaylistSettings: {e}")
        gym_settings = None

    if not gym_settings:
        return {
            "auto_refresh": False,
            "source_playlist_ids": [],
            "last_spotify_playlist_id": None,
        }
    return {
        "auto_refresh": gym_settings.auto_refresh,
        "source_playlist_ids": json.loads(gym_settings.source_playlist_ids or "[]"),
        "last_spotify_playlist_id": gym_settings.last_spotify_playlist_id,
    }


# ── Gym Playlist: Toggle auto-refresh ────────────────
@router.put("/gym-playlist/auto-refresh")
def gym_playlist_toggle_auto_refresh(
    payload: GymPlaylistAutoRefreshRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        gym_settings = (
            db.query(GymPlaylistSettings)
            .filter(GymPlaylistSettings.user_id == current_user.id)
            .first()
        )
        if not gym_settings:
            gym_settings = GymPlaylistSettings(
                user_id=current_user.id,
                auto_refresh=payload.auto_refresh,
                source_playlist_ids="[]",
            )
            db.add(gym_settings)
        else:
            gym_settings.auto_refresh = payload.auto_refresh
        db.commit()
    except Exception as e:
        logger.error(f"Could not save auto-refresh setting: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save setting: {str(e)}",
        )

    return {
        "auto_refresh": gym_settings.auto_refresh,
        "message": "Auto-refresh enabled" if payload.auto_refresh else "Auto-refresh disabled",
    }


# ── Swipe Deck: Gemini-curated from a playlist ───────
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

SWIPE_GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3-flash-preview:generateContent?key={settings.gemini_api_key}"
)

import redis as _redis
_swipe_redis = _redis.Redis.from_url(settings.redis_url, decode_responses=True)
SWIPE_SKIP_TTL = 7 * 86400  # 7 days


def _get_swipe_skips(user_id: int, playlist_id: str) -> set[str]:
    """Get skipped song keys from Redis for this user+playlist."""
    key = f"swipe_skip:{user_id}:{playlist_id}"
    try:
        return set(_swipe_redis.smembers(key))
    except Exception:
        return set()


def _save_swipe_skips(user_id: int, playlist_id: str, skipped: list[str]) -> None:
    """Save skipped songs (title - artist) to Redis set."""
    if not skipped:
        return
    key = f"swipe_skip:{user_id}:{playlist_id}"
    try:
        _swipe_redis.sadd(key, *skipped)
        _swipe_redis.expire(key, SWIPE_SKIP_TTL)
    except Exception as e:
        print(f"SWIPE: Redis save failed: {e}")


@router.get("/discover/swipe", response_model=SwipeDeckResponse)
async def get_swipe_deck(
    playlist_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyse a playlist with Gemini, get 30 new song recommendations, search Spotify."""
    spotify_token = await get_valid_spotify_token(current_user, db)
    headers = {"Authorization": f"Bearer {spotify_token}"}

    try:
        import asyncio, random

        # ── 1. Fetch tracks from the selected playlist ──
        playlist_songs: list[str] = []
        url = f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/items"
        params: dict = {"limit": 50}

        async with httpx.AsyncClient() as client:
            while url and len(playlist_songs) < 200:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code != 200:
                    print(f"SWIPE: playlist items failed: {resp.status_code}")
                    break
                data = resp.json()
                for item in data.get("items", []):
                    track = item.get("track") or item.get("item")
                    if track and track.get("name"):
                        artists = ", ".join(a["name"] for a in track.get("artists", []))
                        playlist_songs.append(f"{track['name']} - {artists}")
                url = data.get("next")
                params = {}

        print(f"SWIPE: Got {len(playlist_songs)} songs from playlist {playlist_id}")

        if len(playlist_songs) < 3:
            raise HTTPException(
                status_code=400,
                detail="This playlist has too few songs. Choose one with at least 3 tracks!",
            )

        # ── 2. Load skip history from Redis ──
        skip_history = _get_swipe_skips(current_user.id, playlist_id)
        print(f"SWIPE: {len(skip_history)} previously skipped songs in history")

        # Build the "avoid" list: playlist songs + skip history
        avoid_songs = playlist_songs.copy()
        if skip_history:
            avoid_songs.extend(list(skip_history)[:50])

        avoid_text = "\n".join(f"- {s}" for s in avoid_songs[:150])

        # ── 3. Ask Gemini for 30 new recommendations ──
        songs_text = "\n".join(f"- {s}" for s in playlist_songs[:100])

        prompt = f"""You are a music recommendation expert. Here is a Spotify playlist:

{songs_text}

Analyze the style, genres and mood of this playlist.
Recommend exactly 30 NEW songs that would be a perfect fit.

Rules:
- Do NOT recommend any of the following songs (already in playlist or skipped):
{avoid_text}
- Mix well-known and lesser-known tracks
- Pay attention to the playlist's genre, mood, energy and language
- Only valid JSON, no markdown, no explanation

Respond ONLY with this JSON format:
{{
  "songs": [
    {{"title": "Song Name", "artist": "Artist Name"}},
    ...
  ]
}}"""

        gemini_payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }

        gemini_songs: list[dict] = []
        for attempt in range(3):
            if attempt > 0:
                await asyncio.sleep(1)
            async with httpx.AsyncClient(timeout=60) as client:
                g_resp = await client.post(SWIPE_GEMINI_URL, json=gemini_payload)
            if g_resp.status_code != 200:
                print(f"SWIPE: Gemini attempt {attempt+1} failed: {g_resp.status_code}")
                continue
            try:
                g_data = g_resp.json()
                g_text = g_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if g_text.startswith("```"):
                    g_text = g_text.split("\n", 1)[1]
                    if g_text.endswith("```"):
                        g_text = g_text[:-3]
                    g_text = g_text.strip()

                # Try normal parse first
                try:
                    parsed = json.loads(g_text)
                    gemini_songs = parsed.get("songs", [])
                except json.JSONDecodeError:
                    # Repair: extract title/artist pairs via regex
                    import re
                    pairs = re.findall(
                        r'"title"\s*:\s*"([^"]+)"\s*,\s*"artist"\s*:\s*"([^"]+)"',
                        g_text,
                    )
                    if pairs:
                        gemini_songs = [{"title": t, "artist": a} for t, a in pairs]
                        print(f"SWIPE: Repaired {len(gemini_songs)} songs from broken JSON")
                    else:
                        raise

                if gemini_songs:
                    break
            except Exception as e:
                print(f"SWIPE: Gemini parse attempt {attempt+1} failed: {e}")
                continue

        print(f"SWIPE: Gemini recommended {len(gemini_songs)} songs")

        if not gemini_songs:
            raise HTTPException(
                status_code=500,
                detail="AI could not generate recommendations. Try again!",
            )

        # ── 4. Search Spotify for each song in parallel (like Discover) ──
        existing_lower = {s.lower() for s in playlist_songs}
        skip_lower = {s.lower() for s in skip_history}

        async def search_one(song: dict) -> dict | None:
            title = song.get("title", "")
            artist = song.get("artist", "")
            query = f"{title} {artist}"

            # Skip if in playlist or skip history
            check = f"{title} - {artist}".lower()
            if check in existing_lower or check in skip_lower:
                return None

            async with httpx.AsyncClient() as client:
                s_resp = await client.get(
                    f"{SPOTIFY_API_BASE}/search",
                    params={"q": query, "type": "track", "limit": 1, "market": "DE"},
                    headers=headers,
                )
            if s_resp.status_code != 200:
                return None

            items = s_resp.json().get("tracks", {}).get("items", [])
            if not items:
                return None

            t = items[0]
            images = t.get("album", {}).get("images", [])
            return {
                "id": t["id"],
                "title": t["name"],
                "artist": ", ".join(a["name"] for a in t.get("artists", [])),
                "album": t.get("album", {}).get("name", ""),
                "album_image": images[0]["url"] if images else None,
                "preview_url": t.get("preview_url"),
                "spotify_uri": t.get("uri", ""),
            }

        results = await asyncio.gather(
            *(search_one(song) for song in gemini_songs)
        )

        # Deduplicate by track ID
        seen_ids: set[str] = set()
        tracks: list[dict] = []
        for r in results:
            if r and r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                tracks.append(r)

        print(f"SWIPE: {len(tracks)} songs found on Spotify")

        if not tracks:
            raise HTTPException(
                status_code=404,
                detail="No new songs found. Try a different playlist!",
            )

        random.shuffle(tracks)
        return {"tracks": tracks[:30]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Swipe Deck failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Swipe Deck could not be loaded: {str(e)}",
        )


# ── Swipe Deck: Save track to a playlist ──────────────
@router.post("/library/save")
async def save_to_playlist(
    payload: SaveTracksRequest,
    playlist_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add tracks to a specific Spotify playlist."""
    spotify_token = await get_valid_spotify_token(current_user, db)

    uris = [f"spotify:track:{tid}" for tid in payload.track_ids]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/items",
            headers={"Authorization": f"Bearer {spotify_token}"},
            json={"uris": uris},
        )

    if resp.status_code not in (200, 201):
        logger.error(f"Save to playlist failed: {resp.status_code} {resp.text[:300]}")
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Could not add song to playlist: {resp.text[:200]}",
        )

    return {"saved": len(payload.track_ids), "already_saved": 0}


# ── Swipe Deck: Report skipped tracks ──────────────────
from pydantic import BaseModel as _BM


class SwipeSkipRequest(_BM):
    songs: list[str]  # "Title - Artist" strings


@router.post("/discover/swipe/skip")
async def report_swipe_skip(
    payload: SwipeSkipRequest,
    playlist_id: str,
    current_user: User = Depends(get_current_user),
):
    """Save skipped songs to Redis so they won't be recommended again."""
    _save_swipe_skips(current_user.id, playlist_id, payload.songs)
    return {"skipped": len(payload.songs)}


# ── Vibe Roast ────────────────────────────────────────
@router.get("/vibe-roast", response_model=RoastResponse)
async def vibe_roast(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a sarcastic AI roast of the user's music taste."""
    spotify_token = await get_valid_spotify_token(current_user, db)

    try:
        result = await generate_vibe_roast(spotify_token)
        return result
    except Exception as e:
        logger.error(f"Vibe Roast failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vibe Roast failed: {str(e)}",
        )


# ── Health check ──────────────────────────────────────
@router.get("/health")
def health_check():
    return {"status": "ok"}

