"""
Configuration module for loading environment variables and settings
"""

import os
from typing import Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_bool_env(key: str, default: bool) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def get_int_env(key: str, default: int) -> int:
    """Get integer value from environment variable"""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_optional_int_env(key: str, default) -> int | None:
    """Get optional integer value from environment variable"""
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Default transcription settings
DEFAULT_TRANSCRIPTION_BACKEND = os.getenv("DEFAULT_TRANSCRIPTION_BACKEND", "openai-whisper")
DEFAULT_WHISPER_MODEL = os.getenv("DEFAULT_WHISPER_MODEL", "large-v3")
DEFAULT_USE_GPU = get_bool_env("DEFAULT_USE_GPU", True)
DEFAULT_GPU_DEVICE = get_optional_int_env("DEFAULT_GPU_DEVICE", None)
DEFAULT_CONVERT_TO_MP3 = get_bool_env("DEFAULT_CONVERT_TO_MP3", True)

# Default LLM settings for clip selection
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2")
DEFAULT_OPEN_ROUTER_MODEL = os.getenv("DEFAULT_OPEN_ROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

# Default clip settings
DEFAULT_MIN_DURATION = get_int_env("DEFAULT_MIN_DURATION", 300)  # 5 minutes in seconds
DEFAULT_MAX_DURATION = get_int_env("DEFAULT_MAX_DURATION", 900)  # 15 minutes in seconds

# Default output settings
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR", "clips")
DEFAULT_TRANSCRIPT_FORMAT = os.getenv("DEFAULT_TRANSCRIPT_FORMAT", "csv")  # Only CSV format supported

# Environment variable names for API keys and settings
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPEN_ROUTER_API_KEY_ENV = "OPEN_ROUTER_API_KEY"
OPEN_ROUTER_MODEL_ENV = "OPEN_ROUTER_MODEL"
GPU_DEVICE_ENV = "VIDEO_CLIPPER_GPU"  # Environment variable to specify GPU device ID


def get_gemini_transcription_api_key() -> Optional[str]:
    """
    Get Gemini API key for transcription.
    Checks GEMINI_TRANSCRIPTION_API_KEY first, then falls back to GEMINI_API_KEY for backwards compatibility.
    """
    key = os.getenv("GEMINI_TRANSCRIPTION_API_KEY")
    if key:
        return key
    # Fallback to legacy key for backwards compatibility
    return os.getenv(GEMINI_API_KEY_ENV)


def get_gemini_llm_api_key() -> Optional[str]:
    """
    Get Gemini API key for LLM operations (clip selection, timestamps, titles).
    Checks GEMINI_LLM_API_KEY first, then falls back to GEMINI_API_KEY for backwards compatibility.
    """
    key = os.getenv("GEMINI_LLM_API_KEY")
    if key:
        return key
    # Fallback to legacy key for backwards compatibility
    return os.getenv(GEMINI_API_KEY_ENV)


def get_openrouter_api_key() -> Optional[str]:
    """Get Open Router API key"""
    return os.getenv(OPEN_ROUTER_API_KEY_ENV)
