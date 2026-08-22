from typing import Literal

from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────
class SpotifyCallback(BaseModel):
    code: str
    redirect_uri: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    spotify_id: str
    email: str | None
    display_name: str | None

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    message: str


# ── Discover ──────────────────────────────────────────
class DiscoverRequest(BaseModel):
    prompt: str
    context_songs: list[str] = []  # e.g. ["Song - Artist", ...] from a playlist
    save_to_playlist: bool = False
    include_my_taste: bool = False


class SongResult(BaseModel):
    title: str
    artist: str
    spotify_url: str | None = None
    album_image: str | None = None
    preview_url: str | None = None
    spotify_uri: str | None = None


class DiscoverResponse(BaseModel):
    songs: list[SongResult]
    mood_summary: str
    playlist_url: str | None = None
    playlist_id: str | None = None
    playlist_name: str | None = None


# ── Playlist ──────────────────────────────────────────
class CreatePlaylistRequest(BaseModel):
    name: str
    description: str = ""
    track_uris: list[str]


class CreatePlaylistResponse(BaseModel):
    playlist_url: str
    playlist_id: str
    name: str
    total_tracks: int


# ── Save Tracks (Lieblingssongs) ──────────────────────
class SaveTracksRequest(BaseModel):
    track_ids: list[str]  # Spotify track IDs (not URIs)


class SaveTracksResponse(BaseModel):
    saved: int
    already_saved: int


# ── Daily Drive ───────────────────────────────────────
class DailyDriveRequest(BaseModel):
    selected_show_ids: list[str] = []  # Spotify show IDs the user picked
    duration_minutes: int = Field(default=60, ge=20, le=240)
    day_mode: Literal["morning", "normal", "night_drive"] = "normal"
    familiarity: int = Field(default=50, ge=0, le=100)  # 0 = familiar, 100 = discoveries


class DailyDriveResponse(BaseModel):
    playlist_url: str
    playlist_id: str
    playlist_name: str
    total_tracks: int
    on_repeat_count: int
    new_discoveries_count: int
    episodes_count: int


# ── Gym Playlist ─────────────────────────────────────
class GymPlaylistGenerateRequest(BaseModel):
    source_playlist_ids: list[str]
    include_on_repeat: bool = False


class GymPlaylistGenerateResponse(BaseModel):
    playlist_url: str
    playlist_id: str
    playlist_name: str
    total_tracks: int
    inspiration_count: int
    auto_refresh: bool


class GymPlaylistSettingsResponse(BaseModel):
    auto_refresh: bool
    source_playlist_ids: list[str]
    include_on_repeat: bool
    last_spotify_playlist_id: str | None


class GymPlaylistAutoRefreshRequest(BaseModel):
    auto_refresh: bool


# ── Swipe Deck ────────────────────────────────────────
class SwipeTrack(BaseModel):
    id: str
    title: str
    artist: str
    album: str
    album_image: str | None = None
    preview_url: str | None = None
    spotify_uri: str


class SwipeDeckResponse(BaseModel):
    tracks: list[SwipeTrack]


# ── Vibe Roast ────────────────────────────────────────
class AudioFeatures(BaseModel):
    danceability: float
    energy: float
    valence: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    tempo: float


class RoastResponse(BaseModel):
    persona: str
    roast: str
    audio_features: AudioFeatures
    top_genres: list[str]
    top_artists: list[str]
    track_count: int
