"""Configuration loaded from the environment / .env file."""
import os

from dotenv import load_dotenv

load_dotenv()


def api_key() -> str:
    """Return the CSFloat API key, or raise if it is not set."""
    key = os.environ.get("CSFLOAT_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CSFLOAT_API_KEY is not set. Add it to a .env file in the project root "
            "(get it from the CSFloat profile → developer tab)."
        )
    return key
