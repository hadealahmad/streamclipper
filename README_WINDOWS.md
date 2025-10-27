# Video Clipper - Windows Setup Guide

AI-powered tool for automatically extracting interesting clips from Arabic videos. Uses OpenAI Whisper for transcription and Gemini for intelligent clip selection.

## System Requirements

- **Windows 10/11** (64-bit)
- **Python 3.9+**
- **FFmpeg** (required for video processing)
- **NVIDIA GPU** (optional but recommended for faster transcription)
- **8GB+ RAM** (16GB recommended for larger models)
- **4GB+ VRAM** (for GPU acceleration)

## Installation Guide

### Step 1: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify installation:
```cmd
python --version
```

### Step 2: Install FFmpeg

FFmpeg is required for video/audio processing. Choose one method:

#### Method A: Using Chocolatey (Recommended)

1. Install Chocolatey if you don't have it:
   - Open PowerShell as Administrator
   - Run: `Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))`

2. Install FFmpeg:
```cmd
choco install ffmpeg
```

#### Method B: Manual Installation

1. Download FFmpeg from [ffmpeg.org](https://www.ffmpeg.org/download.html) (Windows builds)
   - Or use: https://www.gyan.dev/ffmpeg/builds/
   - Download the "release essentials" build

2. Extract the ZIP file to `C:\ffmpeg`

3. Add FFmpeg to system PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Go to "Advanced" tab → "Environment Variables"
   - Under "System variables", find "Path" → "Edit"
   - Click "New" → Add: `C:\ffmpeg\bin`
   - Click OK on all dialogs

4. Verify installation:
```cmd
ffmpeg -version
```

### Step 3: Setup Project

1. Clone or download the project:
```cmd
cd C:\path\to\project
```

2. Create virtual environment:
```cmd
python -m venv venv
```

3. Activate virtual environment:
```cmd
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### Step 4: Install PyTorch (GPU Support)

Install PyTorch based on your GPU:

#### For NVIDIA GPU (CUDA)

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU Only

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify CUDA is available:
```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `True`, GPU support is working.

### Step 5: Install Project Dependencies

```cmd
pip install -r requirements.txt
```

This installs:
- openai-whisper (transcription)
- google-genai (Gemini API)
- python-dotenv (configuration)
- Other dependencies

### Step 6: Configuration

1. Create `.env` file (copy from `env.example` if it exists):
```cmd
copy env.example .env
```

2. Add your Gemini API key to `.env`:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get API key from: https://ai.google.dev/

## Usage

### Basic Commands

```cmd
# Process video with defaults (5-15 min clips)
python clipper.py video.mp4

# Short clips (1-5 minutes)
python clipper.py video.mp4 --min-duration 60 --max-duration 300

# Long-form content (10-30 minutes)
python clipper.py video.mp4 --long-form

# Use specific GPU
python clipper.py video.mp4 --gpu 0

# CPU-only mode
python clipper.py video.mp4 --no-gpu
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to video file | Required |
| `-o, --output` | Output directory | `clips` |
| `-m, --model` | Whisper model size | `medium` |
| `--gpu GPU` | GPU device index | Auto-detect |
| `--no-gpu` | Force CPU usage | False |
| `--min-duration` | Min clip duration (seconds) | `300` |
| `--max-duration` | Max clip duration (seconds) | `900` |
| `--skip-transcribe` | Use existing transcript | False |
| `--skip-analysis` | Use existing clip selections | False |

### Advanced Usage

**Short clips:**
```cmd
python clipper.py video.mp4 --min-duration 60 --max-duration 300
```

**Long-form content:**
```cmd
python clipper.py video.mp4 --long-form
```

**Single-step operations:**
```cmd
python clipper.py video.mp4 --convert-only     # Convert to MP3 only
python clipper.py video.mp4 --transcribe-only  # Transcribe only
python clipper.py video.mp228.1 --analyze-only     # Analyze only
```

## Whisper Models

| Model | Size | Memory | Speed | Use Case |
|-------|------|--------|-------|----------|
| `tiny` | 39 MB | ~1 GB | Fastest | Quick testing |
| `base` | 74 MB | ~1 GB | Fast | Rapid processing |
| `small` | 244 MB | ~2 GB | Medium | Balanced quality |
| `medium` | 769 MB | ~5 GB | Moderate | **Recommended** |
| `large-v3` | 1550 MB | ~10 GB | Slow | Maximum accuracy |

Usage:
```cmd
python clipper.py video.mp4 --model small
```

## GPU Configuration

**Auto-detection (default):**
```cmd
python clipper.py video.mp4
```

**Manual selection:**
```cmd
python clipper.py video.mp4 --gpu 0        # Use GPU 0
python clipper.py video.mp4 --no-gpu       # Force CPU
```

**Environment variable (.env file):**
```
DEFAULT_GPU_DEVICE=0
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

## Troubleshooting

### FFmpeg Not Found

**Error:** `'ffmpeg' is not recognized as an internal or external command`

**Solution:**
1. Verify FFmpeg is installed: `ffmpeg -version`
2. Check PATH environment variable includes FFmpeg
3. Restart command prompt/PowerShell after PATH changes
4. If using manual install, ensure path is: `C:\ffmpeg\bin`

### CUDA Not Available

**Error:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Reinstall PyTorch with CUDA:
   ```cmd
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify NVIDIA GPU drivers are up to date
4. Check GPU is detected by Windows:
   ```cmd
   nvidia-smi
   ```

### Out of Memory

**Error:** `RuntimeError: Out of memory`

**Solutions:**
1. Use smaller Whisper model:
   ```cmd
   python clipper.py video.mp4 --model small
   ```
2. Force CPU mode:
   ```cmd
   python clipper.py video.mp4 --no-gpu
   ```

### Module Not Found

**Error:** `ModuleNotFoundError: No module named '...'`

**Solution:**
```cmd
pip install -r requirements.txt
```

### Virtual Environment Not Activated

**Error:** `python: can't open file 'clipper.py'`

**Solution:**
Always activate virtual environment first:
```cmd
venv\Scripts\activate
```

### Gemini API Key Error

**Error:** `Gemini API key required`

**Solution:**
1. Get API key from: https://ai.google.dev/
2. Add to `.env` file:
   ```
   GEMINI_API_KEY=your_key_here
   ```

### Permission Denied

**Error:** Access denied when installing packages or running scripts

**Solution:**
Run Command Prompt or PowerShell as Administrator

## Configuration File

Create `.env` file with these settings:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Transcription Settings
DEFAULT_TRANSCRIPTION_BACKEND=openai-whisper
DEFAULT_WHISPER_MODEL=medium
DEFAULT_USE_GPU=True
DEFAULT_GPU_DEVICE=0
DEFAULT_CONVERT_TO_MP3=True

# LLM Settings
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_GEMINI_MODEL=gemini-2.5-flash

# Clip Settings
DEFAULT_MIN_DURATION=300
DEFAULT_MAX_DURATION=900

# Output Settings
DEFAULT_OUTPUT_DIR=clips
```

## Performance Tips

1. **Use GPU:** Dramatically faster transcription with NVIDIA GPU
2. **Model Selection:** Use `medium` model for best balance
3. **MP3 Conversion:** Enabled by default for faster processing
4. **Batch Processing:** Process multiple videos by creating a batch script

## Batch Processing Example

Create `process_videos.bat`:

```batch
@echo off
call venv\Scripts\activate.bat
for %%f in (*.mp4) do (
    echo Processing %%f...
    python clipper.py "%%f"
)
pause
```

## Common Workflows

**Complete workflow (new video):**
```cmd
python clipper.py video.mp4
```

**Skip transcription (reuse existing):**
```cmd
python clipper.py video.mp4 --skip-transcribe
```

**Only transcribe:**
```cmd
python clipper.py video.mp4 --transcribe-only
```

**Only analyze existing transcript:**
```cmd
python clipper.py video.mp4 --analyze-only
```

## FAQ

**Q: Do I need a GPU?**  
A: No, but it's 10-20x faster. CPU mode works fine.

**Q: Which GPU should I use?**  
A: Any modern NVIDIA GPU with 4GB+ VRAM. CUDA 12.1 supported.

**Q: How accurate is Arabic transcription?**  
A: Very accurate with `medium` or `large-v3` models.

**Q: How much disk space needed?**  
A: 500MB for models, plus ~200MB per hour of video for clips.

**Q: Can I use AMD GPU?**  
A: No, ROCm (AMD's CUDA) is Linux-only. Use NVIDIA or CPU mode.

## License

MIT License

## Additional Resources

- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Google Gemini API](https://ai.google.dev/)

