# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
EXPORTS_DIR = BASE_DIR / "exports"
MODELS_DIR = BASE_DIR / "models"
BACKUPS_DIR = BASE_DIR / "backups"

# Create directories
for directory in [DATA_DIR, LOGS_DIR, EXPORTS_DIR, MODELS_DIR, BACKUPS_DIR]:
    directory.mkdir(exist_ok=True)

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/crawler.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Crawling settings
DEFAULT_WORKERS = int(os.getenv("DEFAULT_WORKERS", 5))
RATE_LIMIT = float(os.getenv("RATE_LIMIT", 1.0))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", 3))
TIMEOUT = int(os.getenv("TIMEOUT", 10))

# AI/ML settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))

# Search engine settings (if using APIs)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Notification settings
ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "False").lower() == "true"
EMAIL_NOTIFICATIONS = os.getenv("EMAIL_NOTIFICATIONS", "False").lower() == "true"
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")

# Security settings
USER_AGENT = os.getenv("USER_AGENT", "IntelligentCrawler/1.0 (+https://example.com/bot)")
RESPECT_ROBOTS = os.getenv("RESPECT_ROBOTS", "True").lower() == "true"

# Performance settings
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 10000))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", 1000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))

# UI settings
THEME = os.getenv("THEME", "dark" if os.getenv("STREAMLIT_THEME") == "dark" else "light")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 5))  # seconds