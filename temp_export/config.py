from dotenv import load_dotenv
import os
from pathlib import Path


# External secrets file (never committed)
SECRETS_PATH = Path.home() / "secrets" / "job-agent.env"

# --- crawler backfill knobs (temporary) ---
FETCH_PER_PAGE_EXTRACT_LIMIT = 500
FETCH_STOP_AFTER_SEEN_STREAK = 5000
FETCH_MAX_PAGES = 20
FETCH_STOP_IF_PAGE_NEW_ITEMS_LEQ = -1



def load_secrets() -> None:
    """
    Load environment variables from the external secrets file.
    """
    if not SECRETS_PATH.exists():
        raise RuntimeError(f"Secrets file not found at {SECRETS_PATH}")

    load_dotenv(dotenv_path=SECRETS_PATH, override=False)


def require_env(name: str) -> str:
    """
    Return an environment variable or fail loudly if missing.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value
