"""
Default configuration for video clipper
"""

# Default transcription settings
DEFAULT_TRANSCRIPTION_BACKEND = "openai-whisper"  # "faster-whisper", "openai-whisper", "gemini" (openai-whisper better for AMD)
DEFAULT_WHISPER_MODEL = "base"  # "tiny", "base", "small", "medium", "large-v2", "large-v3"
DEFAULT_USE_GPU = True  # Use GPU for local transcription when available

# Default LLM settings
DEFAULT_LLM_PROVIDER = "gemini"  # "gemini", "ollama"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-exp"
DEFAULT_OLLAMA_MODEL = "llama3.2"

# Default clip settings
DEFAULT_MIN_DURATION = 300  # 5 minutes in seconds
DEFAULT_MAX_DURATION = 900  # 15 minutes in seconds

# Default output settings
DEFAULT_OUTPUT_DIR = "clips"
DEFAULT_TRANSCRIPT_FORMAT = "csv"  # "csv" or "json"

# Environment variable names
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
