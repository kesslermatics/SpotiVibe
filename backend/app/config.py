from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/vibeswipe"
    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,https://spotivibe.kesslermatics.com"
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    spotify_redirect_uri: str = "http://127.0.0.1:5173/callback,https://spotivibe.kesslermatics.com/callback"
    openai_api_key: str = ""
    openai_curation_model: str = "gpt-6.1-sol"
    openai_utility_model: str = "gpt-6-luna"
    openai_image_model: str = "gpt-image-2.5-flare"
    redis_url: str = ""

    @property
    def spotify_redirect_uris(self) -> list[str]:
        return [u.strip() for u in self.spotify_redirect_uri.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
