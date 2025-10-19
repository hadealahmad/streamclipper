"""
Default configuration for video clipper
Optimized for AMD GPU systems with OpenAI Whisper
"""

# Default transcription settings
DEFAULT_TRANSCRIPTION_BACKEND = "gemini"  # Primary: openai-whisper (AMD ROCm support)
DEFAULT_WHISPER_MODEL = "medium"  # "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
DEFAULT_USE_GPU = True  # Use GPU for local transcription when available
DEFAULT_GPU_DEVICE = None  # None = auto-detect, 0/1/etc = specific GPU, -1 = force CPU
DEFAULT_CONVERT_TO_MP3 = True  # Convert video to MP3 for faster Whisper processing

# Default LLM settings for clip selection
DEFAULT_LLM_PROVIDER = "gemini"  # Primary: "gemini" (cloud AI), "ollama" (local)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "llama3.2"

# Default clip settings
DEFAULT_MIN_DURATION = 300  # 5 minutes in seconds
DEFAULT_MAX_DURATION = 900  # 15 minutes in seconds

# Default output settings
DEFAULT_OUTPUT_DIR = "clips"
DEFAULT_TRANSCRIPT_FORMAT = "csv"  # Only CSV format supported

# Environment variable names
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GPU_DEVICE_ENV = "VIDEO_CLIPPER_GPU"  # Optional: Set default GPU device

# GPU Configuration
# For AMD GPUs: Install PyTorch with ROCm
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
# For NVIDIA GPUs: Install PyTorch with CUDA
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
