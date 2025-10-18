# Video Clipper

AI-powered tool for automatically extracting interesting clips from Arabic videos using transcription and LLM analysis.

## Features

- **Multiple Transcription Backends**: Gemini (cloud), Faster-Whisper (local), OpenAI Whisper (AMD GPU support)
- **AI-Powered Clip Selection**: Uses LLM to identify interesting segments
- **CSV Export**: Saves transcriptions in CSV format for easy analysis
- **Flexible Duration**: Support for both short clips and long-form content
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

### Advanced Usage

```bash
python video_clipper.py video.mp4 \
    --output clips \
    --backend gemini \
    --llm-provider gemini \
    --min-duration 10 \
    --max-duration 60
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `video` | Path to video file | Required |
| `-o, --output` | Output directory for clips | `clips` |
| `-m, --model` | Whisper model size | `medium` |
| `--backend` | Transcription backend | `faster-whisper` |
| `--gpu` | GPU device index | Auto-detect |
| `--llm-provider` | LLM provider for clip selection | `gemini` |
| `--gemini-api-key` | Gemini API key | From .env |
| `--min-duration` | Minimum clip duration (seconds) | `10` |
| `--max-duration` | Maximum clip duration (seconds) | `60` |
| `--long-form` | Long-form mode (10-30 min clips) | False |
| `--skip-transcribe` | Skip transcription | False |
| `--skip-analysis` | Skip LLM analysis | False |

### Transcription Backends

#### 1. Gemini (Cloud-based)
- **Pros**: Fast, no local model download, high accuracy
- **Cons**: Requires API key, internet connection
- **Usage**: `--backend gemini --llm-provider gemini`

#### 2. Faster-Whisper (Local)
- **Pros**: No API key needed, good performance
- **Cons**: Requires model download, CPU-only
- **Usage**: `--backend faster-whisper`

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
| `base` | 74 MB | Fast | Low | Quick processing |
| `small` | 244 MB | Medium | Medium | Balanced |
| `medium` | 769 MB | Slow | High | Recommended |
| `large-v2` | 1550 MB | Slower | Higher | High accuracy |
| `large-v3` | 1550 MB | Slower | Highest | Best quality |

## Output Files

The tool creates several output files:

- `{video_name}_transcript.csv` - Transcript in CSV format
- `{video_name}_transcript.json` - Transcript in JSON format (backward compatibility)
- `{video_name}_clips.json` - Selected clips metadata
- `{video_name}_clip_01.mp4` - Extracted video clips
- `{video_name}_clip_01.json` - Individual clip metadata

### CSV Format

The transcript CSV contains:
- `start_time` - Start time in seconds
- `end_time` - End time in seconds  
- `text` - Transcribed text

## Examples

### Short Clips (10-60 seconds)
```bash
python video_clipper.py video.mp4 --min-duration 10 --max-duration 60
```

### Long-form Content (10-30 minutes)
```bash
python video_clipper.py video.mp4 --long-form
```

### Using Local Processing
```bash
python video_clipper.py video.mp4 \
    --backend faster-whisper \
    --llm-provider ollama \
    --llm llama3.2
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

- Use `gemini` backend for fastest processing
- Use `faster-whisper` for local processing without API keys
- Use `--long-form` for educational/long content
- Use smaller models for faster processing

## Dependencies

- `faster-whisper` - Local transcription
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