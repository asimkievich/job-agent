from dotenv import load_dotenv
import os
from pathlib import Path

SECRETS_PATH = Path.home() / "secrets" / "job-agent.env"

def load_secrets():
    if not SECRETS_PATH.exists():
        raise RuntimeError(f"Secrets file not found at {SECRETS_PATH}")

    load_dotenv(SECRETS_PATH)

def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value
