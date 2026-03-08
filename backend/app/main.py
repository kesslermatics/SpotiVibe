import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings
from app.database import engine, Base
from app.routes import router
from app.gym_playlist import auto_refresh_gym_playlists

# Configure logging to show INFO and above
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

settings = get_settings()
logger = logging.getLogger(__name__)

# Create all tables on startup (dev convenience – use Alembic for production)
Base.metadata.create_all(bind=engine)

scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: schedule gym playlist auto-refresh at 3:00 AM daily
    scheduler.add_job(
        auto_refresh_gym_playlists,
        trigger=CronTrigger(hour=3, minute=0),
        id="gym_playlist_auto_refresh",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started – Gym Playlist auto-refresh scheduled for 03:00 daily")
    yield
    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("Scheduler shut down")


app = FastAPI(
    title="SpotiVibe API",
    version="0.1.0",
    description="Backend API for SpotiVibe",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────
origins = [o.strip() for o in settings.cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────
app.include_router(router)
