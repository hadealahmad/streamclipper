# Video Clipper

AI-powered tool for automatically extracting interesting clips from Arabic videos using transcription and LLM analysis.

## Features

- **Multiple Transcription Backends**: Gemini (cloud), Faster-Whisper (local with GPU), OpenAI Whisper (AMD GPU support)
- **AI-Powered Clip Selection**: Uses Gemini or Ollama to identify interesting segments
- **Direct CSV Output**: AI outputs CSV format directly for efficient processing
- **Smart Defaults**: Optimized settings for 5-15 minute clips
- **GPU Acceleration**: Automatic GPU detection with CPU fallback
- **Modular Design**: Easily customizable prompts and configuration
- **Arabic Language Support**: Optimized for Arabic (non-formal) content

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd streamclipper
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**:
   - **Arch Linux**: `sudo pacman -S ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

5. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Usage

### Basic Usage

```bash
python video_clipper.py video.mp4
```

The tool now uses smart defaults:
- **Transcription**: Faster-Whisper with GPU acceleration
- **Clip Selection**: Gemini AI
- **Clip Duration**: 5-15 minutes
- **Output**: CSV transcripts + JSON backup

### Advanced Usage

```bash
python video_clipper.py video.mp4 \
    --output clips \
    --backend gemini \
    --llm-provider gemini \
    --min-duration 300 \
    --max-duration 900
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to video file | Required |
| `-o, --output` | Output directory for clips | `clips` |
| `-m, --model` | Whisper model size | `base` |
| `--backend` | Transcription backend | `faster-whisper` |
| `--gpu` | GPU device index | Auto-detect |
| `--no-gpu` | Force CPU usage | False |
| `--llm-provider` | LLM provider for clip selection | `gemini` |
| `--llm` | Ollama model name | `llama3.2` |
| `--gemini-api-key` | Gemini API key | From .env |
| `--min-duration` | Minimum clip duration (seconds) | `300` (5 min) |
| `--max-duration` | Maximum clip duration (seconds) | `900` (15 min) |
| `--long-form` | Long-form mode (10-30 min clips) | False |
| `--transcript-format` | Transcript output format | `csv` |
| `--skip-transcribe` | Skip transcription | False |
| `--skip-analysis` | Skip LLM analysis | False |

### Transcription Backends

#### 1. Gemini (Cloud-based)
- **Pros**: Fast, no local model download, high accuracy
- **Cons**: Requires API key, internet connection
- **Usage**: `--backend gemini --llm-provider gemini`

#### 2. Faster-Whisper (Local with GPU)
- **Pros**: No API key needed, GPU acceleration, good performance
- **Cons**: Requires model download
- **Usage**: `--backend faster-whisper` (GPU auto-detected)
- **Force CPU**: `--no-gpu`

#### 3. OpenAI Whisper (AMD GPU)
- **Pros**: GPU acceleration, high accuracy
- **Cons**: Requires PyTorch with ROCm, larger model files
- **Usage**: `--backend openai-whisper --gpu 0`

### LLM Providers

#### Gemini (Cloud)
- Requires `GEMINI_API_KEY` in `.env`
- Fast and accurate for Arabic content
- Recommended for most users

#### Ollama (Local)
- Requires Ollama installed and running
- No API key needed
- Good for privacy-conscious users

## Configuration

### Customizing Prompts

Edit `prompts.py` to customize AI behavior:
- `GEMINI_TRANSCRIPTION_PROMPT`: Transcription instructions
- `get_clip_selection_prompt()`: Clip selection logic

### Customizing Defaults

Edit `config.py` to change default settings:
- Model sizes, GPU usage, clip durations
- Output directories, transcript formats

### Environment Variables

Create a `.env` file with:

```env
# Required for Gemini backend
GEMINI_API_KEY=your_gemini_api_key_here

# Optional for Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### Model Sizes

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | 39 MB | Fastest | Lowest | Quick testing |
| `base` | 74 MB | Fast | Good | **Default** - Balanced |
| `small` | 244 MB | Medium | Better | Higher accuracy |
| `medium` | 769 MB | Slow | High | Best quality |
| `large-v2` | 1550 MB | Slower | Higher | Maximum accuracy |
| `large-v3` | 1550 MB | Slower | Highest | Best quality |

## Output Files

The tool creates several output files:

- `{video_name}_transcript.csv` - **Primary transcript** (CSV format)
- `{video_name}_transcript.json` - Transcript in JSON format (backward compatibility)
- `{video_name}_clips.json` - Selected clips metadata
- `{video_name}_clip_01.mp4` - Extracted video clips
- `{video_name}_clip_01.json` - Individual clip metadata

### CSV Format (Default)

The transcript CSV contains:
- `start_time` - Start time in seconds
- `end_time` - End time in seconds  
- `text` - Transcribed text

**Benefits of CSV format:**
- Easy to edit in spreadsheet applications
- Direct AI output (no conversion needed)
- Better performance and reliability

## Examples

### Default Usage (5-15 minutes)
```bash
python video_clipper.py video.mp4
```

### Short Clips (1-5 minutes)
```bash
python video_clipper.py video.mp4 --min-duration 60 --max-duration 300
```

### Long-form Content (10-30 minutes)
```bash
python video_clipper.py video.mp4 --long-form
```

### Using Local Processing (No API keys)
```bash
python video_clipper.py video.mp4 \
    --backend faster-whisper \
    --llm-provider ollama \
    --llm llama3.2
```

### Force CPU Usage
```bash
python video_clipper.py video.mp4 --no-gpu
```

### Skip Transcription (Use Existing)
```bash
python video_clipper.py video.mp4 --skip-transcribe
```

### Skip Analysis (Use Existing Clips)
```bash
python video_clipper.py video.mp4 --skip-analysis
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   - Install FFmpeg and ensure it's in your PATH
   - Test with: `ffmpeg -version`

2. **CUDA/GPU issues**:
   - Use `--gpu -1` to force CPU
   - Check GPU drivers and PyTorch installation

3. **API key errors**:
   - Ensure `.env` file exists and contains valid API keys
   - Check API key permissions and quotas

4. **Memory issues**:
   - Use smaller model sizes (`tiny`, `base`, `small`)
   - Process shorter video segments

### Performance Tips

- **Default setup** is optimized for most use cases
- Use `gemini` backend for fastest processing (requires API key)
- Use `faster-whisper` for local processing without API keys
- GPU acceleration is enabled by default (use `--no-gpu` to disable)
- Use `--long-form` for educational/long content
- Use smaller models (`base`, `small`) for faster processing

## Project Structure

```
streamclipper/
├── video_clipper.py      # Main script
├── prompts.py            # AI prompts (user-editable)
├── config.py             # Configuration defaults
├── clips/                # Default output directory
└── README.md             # This file
```

### Modular Design

- **`prompts.py`**: Customize AI behavior by editing prompts
- **`config.py`**: Change default settings without touching main code
- **`video_clipper.py`**: Core functionality and logic

## Dependencies

- `faster-whisper` - Local transcription with GPU support
- `google-generativeai` - Gemini API
- `requests` - HTTP requests
- `tqdm` - Progress bars
- `python-dotenv` - Environment variables

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information