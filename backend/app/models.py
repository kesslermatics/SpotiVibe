from sqlalchemy import Column, Integer, String, Boolean, Text, ForeignKey
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    spotify_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    display_name = Column(String, nullable=True)
    spotify_access_token = Column(String, nullable=True)
    spotify_refresh_token = Column(String, nullable=True)


class GymPlaylistSettings(Base):
    __tablename__ = "gym_playlist_settings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    auto_refresh = Column(Boolean, default=False, nullable=False)
    source_playlist_ids = Column(Text, default="[]", nullable=False)  # JSON array
    last_spotify_playlist_id = Column(String, nullable=True)


class DailyWalkSettings(Base):
    __tablename__ = "daily_walk_settings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    auto_refresh = Column(Boolean, default=False, nullable=False)
    selected_show_ids = Column(Text, default="[]", nullable=False)   # JSON array of Spotify show IDs
    last_spotify_playlist_id = Column(String, nullable=True)
    # Settings persisted so the scheduler can regenerate without user input
    duration_minutes = Column(Integer, default=45, nullable=False)
    familiarity = Column(Integer, default=50, nullable=False)        # 0–100
