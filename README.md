# Video Clipper

AI-powered tool for automatically extracting interesting clips from Arabic videos. Uses OpenAI Whisper for transcription and Gemini for intelligent clip selection.

## Project Structure

This project uses a feature-based architecture with clear separation of concerns:

```
streamclipper/
├── clipper.py              # Main entry point
├── prompts.py              # LLM prompts configuration
├── src/                    # Feature modules
│   ├── config.py           # Configuration management
│   ├── utils/              # GPU detection, dependencies, file I/O
│   ├── transcription/      # Transcription backends (Whisper, Gemini)
│   ├── analysis/           # LLM-based analysis tools
│   │   └── llm/           # LLM providers (Gemini, Ollama, OpenRouter)
│   └── extraction/         # Video clipping and audio conversion
└── clips/                  # Output directory
```

See `REFACTOR_STATUS.md` for details on the refactoring.

## Features

- **GPU Acceleration**: AMD (ROCm), NVIDIA (CUDA), or CPU
- **Multi-GPU Support**: Interactive GPU selection
- **OpenAI Whisper**: High-quality Arabic transcription
- **Gemini AI**: Intelligent clip selection
- **Smart Defaults**: 5-15 minute clips
- **Flexible Configuration**: Multiple backends

## Quick Start

### 1. Install Dependencies

```bash
# Install FFmpeg
sudo pacman -S ffmpeg  # Arch
sudo apt install ffmpeg  # Ubuntu/Debian

# Create virtual environment
python -m venv venv

source venv/bin/activate
or
source venv/bin/activate.fish

# Install PyTorch (AMD GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here
```

### 3. Run

```bash
python clipper.py video.mp4
```

## Usage

### Basic Commands

```bash
# Process video with defaults set in the dotenv (5-15 min clips)
python clipper.py video.mp4

# Short clips (1-5 minutes)
python clipper.py video.mp4 --min-duration 60 --max-duration 300

# Long-form content (10-30 minutes)
python clipper.py video.mp4 --long-form

# Use specific GPU
python clipper.py video.mp4 --gpu 1

# CPU-only mode
python clipper.py video.mp4 --no-gpu
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to video file | Required |
| `-o, --output` | Output directory for clips | `clips` |
| `-m, --model` | Whisper model size | `medium` |
| `--backend` | Transcription backend | `openai-whisper` |
| `--gpu GPU` | GPU device index (0, 1, etc) | Auto-detect |
| `--interactive-gpu` | Interactive GPU selection | False |
| `--no-gpu` | Force CPU usage | False |
| `--llm-provider` | LLM provider for clip selection | `gemini` |
| `--llm` | Ollama/OpenRouter model name | `llama3.2` |
| `--gemini-api-key` | Gemini API key | From `.env` |
| `--openrouter-api-key` | Open Router API key | From `.env` |
| `--openrouter-model` | Open Router model name | From `.env` |
| `--min-duration` | Minimum clip duration (seconds) | `300` |
| `--max-duration` | Maximum clip duration (seconds) | `900` |
| `--long-form` | Long-form mode (10-30 min) | False |
| `--skip-transcribe` | Use existing transcript | False |
| `--skip-analysis` | Use existing clip selections | False |
| `--no-mp3` | Skip MP3 conversion | False |
| `--convert-only` | Only convert video to MP3 | False |
| `--transcribe-only` | Only transcribe video | False |
| `--analyze-only` | Only analyze existing transcript | False |
| `--extract-only` | Only extract clips from existing analysis | False |
| `--cleanup-audio` | Clean up MP3 files after processing | False |
| `--force-redo` | Force redo all steps even if files exist | False |

### GPU Configuration

**Auto-detection (default):**
```bash
python clipper.py video.mp4  # Uses first available GPU
```

**Manual selection:**
```bash
python clipper.py video.mp4 --gpu 1        # Use GPU 1
python clipper.py video.mp4 --no-gpu       # Force CPU
python clipper.py video.mp4 --interactive-gpu  # Interactive selection
```

**Environment variable:**
```bash
# In .env file
VIDEO_CLIPPER_GPU=1
```

### Advanced Usage

**Short clips (1-5 minutes):**
```bash
python clipper.py video.mp4 --min-duration 60 --max-duration 300
```

**Long-form content (10-30 minutes):**
```bash
python clipper.py video.mp4 --long-form
```

**Skip steps:**
```bash
python clipper.py video.mp4 --skip-transcribe  # Use existing transcript
python clipper.py video.mp4 --skip-analysis    # Use existing clips
```

**Single-step operations:**
```bash
python clipper.py video.mp4 --convert-only     # Convert to MP3 only
python clipper.py video.mp4 --transcribe-only  # Transcribe only
python clipper.py video.mp4 --analyze-only     # Analyze existing transcript
python clipper.py video.mp4 --extract-only     # Extract clips only
```

**File management:**
```bash
python clipper.py video.mp4 --force-redo       # Redo all steps
python clipper.py video.mp4 --cleanup-audio    # Remove MP3 files
```

### Whisper Models

| Model | Size | Memory | Speed | Accuracy | Use Case |
|-------|------|--------|-------|----------|----------|
| `tiny` | 39 MB | ~1 GB | Fastest | Basic | Quick testing |
| `base` | 74 MB | ~1 GB | Fast | Good | Rapid processing |
| `small` | 244 MB | ~2 GB | Medium | Better | Balanced quality |
| `medium` | 769 MB | ~5 GB | Moderate | High | **Recommended** |
| `large` | 1550 MB | ~10 GB | Slow | Highest | Maximum accuracy |
| `large-v2` | 1550 MB | ~10 GB | Slow | Highest | Alternative large |
| `large-v3` | 1550 MB | ~10 GB | Slow | Highest | Latest large |

### Backends

**OpenAI Whisper (default):**
- Excellent AMD GPU support via ROCm
- High accuracy for Arabic content
- Multi-GPU support

**Faster-Whisper (optional):**
- Lower memory usage
- Faster inference
- Install: `pip install faster-whisper`

**Gemini (cloud-based):**
- No local model download
- No GPU required
- Requires API key

### LLM Providers

**Gemini (default):**
```bash
# Add to .env
GEMINI_API_KEY=your_api_key
```

**Open Router (cloud):**
```bash
# Add to .env
OPEN_ROUTER_API_KEY=your_api_key
OPEN_ROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free

# Use
python clipper.py video.mp4 --llm-provider openrouter
```

**Ollama (local):**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
ollama serve

# Use
python clipper.py video.mp4 --llm-provider ollama --llm llama3.2
```

## Output Files

```
clips/
├── video_name_audio.mp3           # Audio file
├── video_name_transcript.csv      # Transcript with timestamps
├── video_name_clips.json          # Clip selections metadata
├── video_name_clip_01.mp4         # Extracted video clip 1
└── video_name_clip_01.json        # Clip 1 metadata
```

## Configuration

Edit `.env` to customize defaults:

```bash
# Video Clipper Configuration
# Copy this file to .env and customize as needed

# ============================================================
# API Keys
# ============================================================

# Required for Gemini LLM provider and Gemini transcription backend
GEMINI_API_KEY=your_gemini_api_key_here

# ============================================================
# Transcription Settings
# ============================================================

# Transcription backend: "openai-whisper", "faster-whisper", or "gemini"
# Default: openai-whisper (AMD ROCm support)
DEFAULT_TRANSCRIPTION_BACKEND=openai-whisper

# Whisper model size: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
# Larger models are more accurate but slower and use more memory
# Default: large-v3
DEFAULT_WHISPER_MODEL=large-v3

# Use GPU for transcription when available
# Default: True
DEFAULT_USE_GPU=True

# GPU device selection
# None = auto-detect first GPU
# 0/1/etc = specific GPU index
# -1 = force CPU mode
# Default: None (auto-detect)
DEFAULT_GPU_DEVICE=

# Optional: Set default GPU device (alternative to DEFAULT_GPU_DEVICE)
# This is the legacy environment variable name
VIDEO_CLIPPER_GPU=

# Convert video to MP3 before transcription for faster processing
# Default: True
DEFAULT_CONVERT_TO_MP3=True

# ============================================================
# LLM Settings (for clip selection)
# ============================================================

# LLM provider: "gemini" (cloud AI), "ollama" (local), or "openrouter" (cloud AI router)
# Default: gemini
DEFAULT_LLM_PROVIDER=gemini

# Gemini model to use for clip selection
# Examples: "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"
# Default: gemini-2.5-flash
DEFAULT_GEMINI_MODEL=gemini-2.5-flash

# Ollama model to use when LLM provider is "ollama"
# Examples: "llama3.2", "llama3.2:70b", "mistral"
# Default: llama3.2
DEFAULT_OLLAMA_MODEL=llama3.2

# Open Router API key (required when using openrouter provider)
# Get your key from: https://openrouter.ai/
OPEN_ROUTER_API_KEY=

# Open Router model to use for clip selection
# Examples: "meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-4", "google/gemini-pro", "mistralai/mistral-large"
# Default: meta-llama/llama-3.3-70b-instruct:free
# See available models: https://openrouter.ai/models
OPEN_ROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free

# ============================================================
# Clip Settings
# ============================================================

# Minimum clip duration in seconds
# Default: 300 (5 minutes)
DEFAULT_MIN_DURATION=300

# Maximum clip duration in seconds
# Default: 900 (15 minutes)
DEFAULT_MAX_DURATION=900

# ============================================================
# Output Settings
# ============================================================

# Output directory for clips and transcripts
# Default: clips
DEFAULT_OUTPUT_DIR=clips

# Transcript format (only CSV is currently supported)
# Default: csv
DEFAULT_TRANSCRIPT_FORMAT=csv

# ============================================================
# GPU Configuration Notes
# ============================================================

# For AMD GPUs: Install PyTorch with ROCm
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# For NVIDIA GPUs: Install PyTorch with CUDA
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only: Install PyTorch CPU version
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

```

## Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Verify PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

**Out of memory:**
```bash
# Use smaller model
python clipper.py video.mp4 --model small

# Force CPU
python clipper.py video.mp4 --no-gpu
```

**API key error:**
```bash
# Create .env file for Gemini
echo "GEMINI_API_KEY=your_key" > .env

# Or for Open Router
echo "OPEN_ROUTER_API_KEY=your_key" >> .env

# Or use local processing
python clipper.py video.mp4 --llm-provider ollama
```

## FAQ

**Q: Which GPU to use?**  
A: Use `--interactive-gpu` to choose, or `--gpu 0/1` to specify.

**Q: Can I use without GPU?**  
A: Yes, use `--no-gpu` for CPU mode.

**Q: How accurate is Arabic transcription?**  
A: Very accurate with `medium` or larger models.

**Q: How much VRAM needed?**  
A: 4GB for `medium`, 8GB for `large`.

## License

MIT License
