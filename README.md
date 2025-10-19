# Video Clipper

AI-powered tool for automatically extracting interesting clips from Arabic videos. Optimized for AMD GPUs with ROCm support, using OpenAI Whisper for transcription and Gemini for intelligent clip selection.

## Overview

Video Clipper analyzes video content using AI transcription and natural language understanding to automatically identify and extract compelling segments. Perfect for content creators working with long-form Arabic video content.

### Key Features

- **AMD GPU Acceleration**: Primary focus on AMD GPUs via ROCm with automatic detection
- **Dual/Multi-GPU Support**: Interactive GPU selection for systems with multiple GPUs
- **OpenAI Whisper**: High-quality speech-to-text transcription (primary backend)
- **Gemini AI**: Intelligent clip selection based on content analysis
- **Arabic Language**: Optimized for informal Arabic (عامية) with mixed English
- **Smart Defaults**: Pre-configured for 5-15 minute clips
- **Flexible Configuration**: Multiple backends and customizable prompts

### Architecture

```
Video Input
    ↓
Audio Extraction (ffmpeg → MP3)
    ↓
Transcription (OpenAI Whisper on AMD GPU)
    ↓
Transcript (CSV format with timestamps)
    ↓
AI Analysis (Gemini API)
    ↓
Clip Selection (JSON metadata)
    ↓
Video Extraction (ffmpeg)
    ↓
Final Clips + Metadata
```

## Installation

### System Requirements

- **Operating System**: Linux (recommended for AMD GPU), Windows, macOS
- **Python**: 3.8 or higher
- **GPU**: AMD (ROCm 5.6+), NVIDIA (CUDA 11.8+), or CPU
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2-5GB for Whisper models

### Prerequisites

#### 1. Install FFmpeg

FFmpeg is required for video/audio processing:

```bash
# Arch Linux
sudo pacman -S ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

Verify installation:
```bash
ffmpeg -version
ffprobe -version
```

#### 2. Install Python

```bash
# Check Python version (3.8+ required)
python3 --version

# Install pip if needed
sudo pacman -S python-pip  # Arch
sudo apt install python3-pip  # Ubuntu/Debian
```

### Setup Instructions

#### 1. Clone Repository

```bash
git clone <repository-url>
cd streamclipper
```

#### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

#### 3. Install PyTorch with GPU Support

**For AMD GPUs (ROCm 6.4):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

**For NVIDIA GPUs (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify PyTorch installation:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `openai-whisper` - Speech-to-text transcription
- `google-generativeai` - Gemini API for clip selection
- `python-dotenv` - Environment variable management
- `requests` - HTTP client
- `tqdm` - Progress bars

#### 5. Configure API Keys

Create a `.env` file in the project root:

```bash
# Required for Gemini backend
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Set default GPU device
VIDEO_CLIPPER_GPU=0
```

**Get Gemini API Key:**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create or select a project
3. Generate API key
4. Add to `.env` file

## Usage

### Basic Usage

Process a video with default settings (5-15 minute clips):

```bash
python video_clipper.py video.mp4
```

Default behavior:
- **Transcription**: OpenAI Whisper with GPU acceleration
- **Clip Selection**: Gemini AI
- **Clip Duration**: 5-15 minutes
- **GPU**: Auto-detect and use first GPU
- **Output**: `clips/` directory

### Command Line Options

```
usage: video_clipper.py [-h] [-o OUTPUT] [-m MODEL] [--backend BACKEND]
                        [--gpu GPU] [--interactive-gpu] [--no-gpu]
                        [--llm-provider LLM_PROVIDER] [--llm LLM]
                        [--gemini-api-key GEMINI_API_KEY]
                        [--min-duration MIN_DURATION]
                        [--max-duration MAX_DURATION] [--long-form]
                        [--skip-transcribe] [--skip-analysis] [--no-mp3]
                        video
```

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
| `--llm` | Ollama model name | `llama3.2` |
| `--gemini-api-key` | Gemini API key | From `.env` |
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

### Whisper Model Sizes

| Model | Size | Memory | Speed | Accuracy | Use Case |
|-------|------|--------|-------|----------|----------|
| `tiny` | 39 MB | ~1 GB | Fastest | Basic | Quick testing |
| `base` | 74 MB | ~1 GB | Fast | Good | Rapid processing |
| `small` | 244 MB | ~2 GB | Medium | Better | Balanced quality |
| `medium` | 769 MB | ~5 GB | Moderate | High | **Recommended** |
| `large` | 1550 MB | ~10 GB | Slow | Highest | Maximum accuracy |
| `large-v2` | 1550 MB | ~10 GB | Slow | Highest | Alternative large |
| `large-v3` | 1550 MB | ~10 GB | Slow | Highest | Latest large |

### GPU Configuration

#### Automatic GPU Detection

By default, the script auto-detects and uses the first available GPU:

```bash
python video_clipper.py video.mp4
```

Output:
```
============================================================
GPU Detection - ROCM Backend
============================================================

Found 2 GPU(s):

  [0] AMD Radeon RX 7900 XTX
      Memory: 24.0 GB
      Compute: 11.0

  [1] AMD Radeon RX 6800
      Memory: 16.0 GB
      Compute: 10.3

Auto-selected GPU 0: AMD Radeon RX 7900 XTX
```

#### Manual GPU Selection

Select a specific GPU:

```bash
# Use GPU 1
python video_clipper.py video.mp4 --gpu 1

# Force CPU mode
python video_clipper.py video.mp4 --gpu -1
# or
python video_clipper.py video.mp4 --no-gpu
```

#### Interactive GPU Selection

For multi-GPU systems, use interactive mode:

```bash
python video_clipper.py video.mp4 --interactive-gpu
```

This prompts you to select which GPU to use.

#### Environment Variable

Set a default GPU device:

```bash
# In .env file
VIDEO_CLIPPER_GPU=1

# Or export for session
export VIDEO_CLIPPER_GPU=1
```

### Advanced Examples

#### Short Clips (1-5 minutes)

```bash
python video_clipper.py video.mp4 \
    --min-duration 60 \
    --max-duration 300
```

#### Long-form Content (10-30 minutes)

```bash
python video_clipper.py video.mp4 --long-form
```

This automatically sets min=600s (10 min) and max=1800s (30 min).

#### Specific GPU with Custom Model

```bash
python video_clipper.py video.mp4 \
    --gpu 1 \
    --model large-v2 \
    --min-duration 180 \
    --max-duration 600
```

#### Use Existing Transcript

Skip transcription if you already have a transcript:

```bash
python video_clipper.py video.mp4 --skip-transcribe
```

This loads `clips/{video_name}_transcript.csv`.

#### Use Existing Clip Selections

Skip AI analysis if you have clip selections:

```bash
python video_clipper.py video.mp4 --skip-analysis
```

This loads `clips/{video_name}_clips.json`.

### Single-Step Operations

#### Convert Video to MP3 Only

```bash
python video_clipper.py video.mp4 --convert-only
```

This only converts the video to MP3 format for faster processing later.

#### Transcribe Only

```bash
python video_clipper.py video.mp4 --transcribe-only
```

This only transcribes the video and saves the transcript files.

#### Analyze Existing Transcript

```bash
python video_clipper.py video.mp4 --analyze-only
```

This analyzes an existing transcript file to select clips.

#### Extract Clips Only

```bash
python video_clipper.py video.mp4 --extract-only
```

This extracts clips from existing analysis without re-transcribing or re-analyzing.

### File Management

#### Force Redo All Steps

```bash
python video_clipper.py video.mp4 --force-redo
```

This forces redoing all steps even if intermediate files exist.

#### Clean Up Audio Files

```bash
python video_clipper.py video.mp4 --cleanup-audio
```

This removes MP3 files after processing is complete.

#### Local Processing (No API Keys)

Use Ollama instead of Gemini:

```bash
# Install Ollama first: https://ollama.ai
ollama pull llama3.2

python video_clipper.py video.mp4 \
    --backend openai-whisper \
    --llm-provider ollama \
    --llm llama3.2
```

#### CPU-only Mode

Force CPU if no GPU available or for testing:

```bash
python video_clipper.py video.mp4 --no-gpu
```

### Transcription Backends

#### 1. OpenAI Whisper (Default, Recommended for AMD)

**Advantages:**
- Excellent AMD GPU support via ROCm
- High accuracy for Arabic content
- Multi-GPU support with device selection
- Mature and well-tested

**Requirements:**
- PyTorch with ROCm or CUDA
- openai-whisper package

**Usage:**
```bash
python video_clipper.py video.mp4 --backend openai-whisper
```

#### 2. Faster-Whisper (Optional)

**Advantages:**
- Lower memory usage
- Faster inference on some systems
- Good for limited hardware

**Disadvantages:**
- Less robust GPU support
- May require additional setup

**Requirements:**
- faster-whisper package (install separately)

**Usage:**
```bash
pip install faster-whisper
python video_clipper.py video.mp4 --backend faster-whisper
```

#### 3. Gemini (Cloud-based)

**Advantages:**
- No local model download
- Fast processing
- No GPU required
- High accuracy
- Automatic MP3 conversion for faster uploads

**Disadvantages:**
- Requires API key and internet
- API usage costs
- Upload time (reduced with MP3 conversion)

**Usage:**
```bash
python video_clipper.py video.mp4 --backend gemini
```

**Note:** Gemini backend automatically converts videos to MP3 before upload to reduce file size and upload time. Gemini supports various audio formats including MP3, WAV, AAC, OGG, and FLAC.

### LLM Providers for Clip Selection

#### Gemini (Default, Recommended)

Cloud-based AI for intelligent clip selection.

**Setup:**
```bash
# Add to .env
GEMINI_API_KEY=your_api_key
```

**Usage:**
```bash
python video_clipper.py video.mp4 --llm-provider gemini
```

#### Ollama (Local, Privacy-focused)

Run LLM locally without API keys.

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2

# Start server
ollama serve
```

**Usage:**
```bash
python video_clipper.py video.mp4 \
    --llm-provider ollama \
    --llm llama3.2
```

## Output Files

The tool generates organized output in the specified directory:

```
clips/
├── video_name_audio.mp3           # Audio file (kept for reuse)
├── video_name_transcript.csv      # Primary transcript (CSV with timestamps)
├── video_name_transcript.txt      # Plain text transcript (no timestamps)
├── video_name_clips.json          # Clip selections metadata
├── video_name_clip_01.mp4         # Extracted video clip 1
├── video_name_clip_01.json        # Clip 1 metadata
├── video_name_clip_02.mp4         # Extracted video clip 2
├── video_name_clip_02.json        # Clip 2 metadata
└── ...
```

### Transcript Format (CSV)

```csv
start_time,end_time,text
0.0,5.2,النص العربي هنا
5.5,12.3,المزيد من النص
```

Fields:
- `start_time`: Start time in seconds (decimal)
- `end_time`: End time in seconds (decimal)
- `text`: Transcribed text (Arabic/English mixed)

### Plain Text Transcript Format (TXT)

```text
النص العربي هنا
المزيد من النص
```

This file contains only the transcribed text without timestamps, useful for:
- Reading the full content
- Copy-pasting text
- Further text processing

### Clips Metadata (JSON)

```json
[
  {
    "start": 10.5,
    "end": 635.2,
    "reason": "مناقشة كاملة حول الموضوع"
  }
]
```

### Individual Clip Metadata

```json
{
  "clip_number": 1,
  "start": 10.5,
  "end": 635.2,
  "duration": 624.7,
  "reason": "مناقشة كاملة حول الموضوع",
  "source_video": "/path/to/video.mp4"
}
```

## Configuration

### Customizing Prompts

Edit `prompts.py` to change AI behavior:

```python
# Transcription prompt (for Gemini backend)
GEMINI_TRANSCRIPTION_PROMPT = """..."""

# Clip selection prompt generator
def get_clip_selection_prompt(transcript_text, min_duration, max_duration):
    """Customize clip selection logic"""
    ...
```

### Customizing Defaults

Edit `config.py` to change default settings:

```python
DEFAULT_TRANSCRIPTION_BACKEND = "openai-whisper"
DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_USE_GPU = True
DEFAULT_GPU_DEVICE = None  # Auto-detect

DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_MIN_DURATION = 300  # 5 minutes
DEFAULT_MAX_DURATION = 900  # 15 minutes
```

## Troubleshooting

### GPU Issues

#### GPU Not Detected

**Problem:** Script shows "No GPU detected"

**Solutions:**

1. **Verify PyTorch installation:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

2. **For AMD GPUs, install ROCm PyTorch:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

3. **Check ROCm installation:**
```bash
rocm-smi  # Should list your AMD GPUs
```

4. **Verify GPU drivers:**
```bash
lspci | grep -i vga  # List GPUs
```

#### Wrong GPU Selected

**Problem:** Script uses wrong GPU on multi-GPU system

**Solutions:**

1. **Manual selection:**
```bash
python video_clipper.py video.mp4 --gpu 1
```

2. **Interactive selection:**
```bash
python video_clipper.py video.mp4 --interactive-gpu
```

3. **Set environment variable:**
```bash
export VIDEO_CLIPPER_GPU=1
```

#### Out of Memory Errors

**Problem:** "CUDA out of memory" or similar

**Solutions:**

1. **Use smaller model:**
```bash
python video_clipper.py video.mp4 --model small
```

2. **Force CPU:**
```bash
python video_clipper.py video.mp4 --no-gpu
```

3. **Close other GPU applications**

4. **Process shorter videos**

### Installation Issues

#### FFmpeg Not Found

**Problem:** "ffmpeg is NOT installed"

**Solution:**
```bash
# Arch Linux
sudo pacman -S ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

#### API Key Errors

**Problem:** "Gemini API key required"

**Solutions:**

1. **Create `.env` file:**
```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

2. **Use command line:**
```bash
python video_clipper.py video.mp4 --gemini-api-key YOUR_KEY
```

3. **Use local processing:**
```bash
python video_clipper.py video.mp4 --llm-provider ollama
```

#### Import Errors

**Problem:** "ModuleNotFoundError: No module named 'whisper'"

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# Install other dependencies
pip install -r requirements.txt
```

### ROCm-Specific Issues

#### ROCm Not Detected

**Problem:** PyTorch doesn't detect AMD GPU

**Solutions:**

1. **Install ROCm drivers:**
```bash
# Arch Linux
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk

# Ubuntu (requires repo setup)
sudo apt install rocm-hip-sdk rocm-opencl-sdk
```

2. **Add user to render group:**
```bash
sudo usermod -a -G render,video $USER
# Logout and login again
```

3. **Set environment variables:**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Adjust for your GPU
```

4. **Verify ROCm:**
```bash
rocm-smi
/opt/rocm/bin/rocminfo
```

#### Compute Capability Issues

**Problem:** "Your GPU compute capability is not supported"

**Solution:**

ROCm supports:
- RDNA 2 (gfx10.3): RX 6000 series
- RDNA 3 (gfx11.0): RX 7000 series
- CDNA 2 (gfx9.0): MI200 series

Check compatibility: https://rocm.docs.amd.com/

### Performance Issues

#### Slow Transcription

**Solutions:**

1. **Enable GPU:**
```bash
python video_clipper.py video.mp4 --gpu 0
```

2. **Use smaller model:**
```bash
python video_clipper.py video.mp4 --model base
```

3. **Enable MP3 conversion (default):**
```bash
python video_clipper.py video.mp4  # MP3 enabled by default
```

4. **Use faster backend:**
```bash
pip install faster-whisper
python video_clipper.py video.mp4 --backend faster-whisper
```

#### Slow Clip Selection

**Solution:**

Gemini is already fast. For local processing:
```bash
ollama pull llama3.2:70b  # Larger model = better but slower
python video_clipper.py video.mp4 --llm-provider ollama --llm llama3.2:70b
```

## Development

### Project Structure

```
streamclipper/
├── video_clipper.py       # Main application
├── config.py              # Configuration defaults
├── prompts.py             # AI prompts (user-editable)
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
├── REFACTORING.md         # Refactoring plan
├── .env                   # API keys (create from template)
├── clips/                 # Default output directory
└── venv/                  # Virtual environment
```

### Code Architecture

**Key Classes:**

- `GPUDetector`: GPU detection and selection
- `WhisperTranscriber`: OpenAI Whisper backend (primary)
- `FasterWhisperTranscriber`: Faster-whisper backend (optional)
- `GeminiTranscriber`: Gemini transcription backend
- `GeminiLLM`: Gemini clip selection
- `OllamaLLM`: Local LLM clip selection
- `ClipSelector`: Clip analysis and selection
- `VideoClipper`: Video extraction with ffmpeg
- `AudioConverter`: MP3 conversion utilities
- `DependencyChecker`: Dependency validation

### Extending the Tool

#### Add New Transcription Backend

1. Create a new transcriber class:
```python
class NewTranscriber:
    def __init__(self, ...):
        pass
    
    def transcribe(self, video_path: str) -> List[Dict]:
        # Return list of {'start': float, 'end': float, 'text': str}
        pass
```

2. Add to `main()` backend selection

3. Update `config.py` and documentation

#### Add New LLM Provider

1. Create a new LLM class:
```python
class NewLLM:
    def __init__(self, ...):
        pass
    
    def generate(self, prompt: str) -> str:
        # Return generated text
        pass
```

2. Add to `main()` LLM provider selection

3. Update configuration

## Performance Tips

1. **Use GPU acceleration** - 5-10x faster than CPU
2. **Choose appropriate model size** - `medium` is balanced
3. **Enable MP3 conversion** - Faster Whisper processing (enabled by default)
4. **Use Gemini for clip selection** - Fast and accurate
5. **Select correct GPU** - Use `--interactive-gpu` on multi-GPU systems
6. **Process in batches** - For multiple videos
7. **Skip completed steps** - Use `--skip-transcribe` and `--skip-analysis`

## Best Practices

### For AMD GPU Users

1. **Install ROCm PyTorch first:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```

2. **Verify GPU detection:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

3. **Use `medium` or `large` models** - AMD GPUs have good VRAM

4. **Monitor GPU usage:**
```bash
watch -n 1 rocm-smi
```

### For Multi-GPU Systems

1. **Use interactive selection:**
```bash
python video_clipper.py video.mp4 --interactive-gpu
```

2. **Set environment variable:**
```bash
export VIDEO_CLIPPER_GPU=1  # Use GPU 1
```

3. **Run multiple instances** - One per GPU for parallel processing

### For Long Videos

1. **Use long-form mode:**
```bash
python video_clipper.py video.mp4 --long-form
```

2. **Increase max duration:**
```bash
python video_clipper.py video.mp4 --max-duration 1800  # 30 minutes
```

3. **Process in stages** - Transcribe once, analyze multiple times

## FAQ

**Q: Which GPU should I use on a dual GPU system?**

A: Use `--interactive-gpu` to see details and choose, or `--gpu 0/1` to specify. Generally, use the more powerful GPU.

**Q: Can I use this without a GPU?**

A: Yes, use `--no-gpu` for CPU mode. It's slower but works.

**Q: Do I need a Gemini API key?**

A: Only for Gemini backend. Use `--llm-provider ollama` for local processing.

**Q: How accurate is the Arabic transcription?**

A: Very accurate with `medium` or larger models. Whisper is trained on Arabic.

**Q: Can I customize what clips are selected?**

A: Yes, edit `prompts.py` to change selection criteria.

**Q: How much VRAM do I need?**

A: 4GB for `medium`, 8GB for `large`. See model table above.

**Q: Can I process multiple videos?**

A: Yes, use a shell loop or run multiple instances.

**Q: How do I update the tool?**

A: `git pull` and `pip install -r requirements.txt --upgrade`

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

- **Issues**: Open an issue on GitHub
- **Documentation**: This README
- **Configuration**: See `config.py` and `prompts.py`

## Acknowledgments

- OpenAI Whisper for transcription
- Google Gemini for AI analysis
- AMD ROCm for GPU acceleration
- FFmpeg for video processing

---

**Made with ❤️ for Arabic content creators**
