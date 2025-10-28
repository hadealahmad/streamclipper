# Context-Aware Dependency Checking

## Summary

Implemented smart dependency checking that only verifies required dependencies based on the actual command being executed.

## Changes Made

### 1. Updated `src/utils/dependencies.py`

Added `check_and_install_specific(ctx)` method that:
- Checks ffmpeg/ffprobe only if video processing is needed
- Checks Whisper/PyTorch only if using local transcription
- Checks google-genai only if using Gemini
- Checks Ollama connection only if using Ollama LLM
- Checks OpenRouter only if using OpenRouter LLM

### 2. Updated `clipper.py`

Added `build_dependency_context(args)` function that:
- Analyzes CLI arguments to determine which features are being used
- Returns a context dict specifying which dependencies are actually needed
- Handles single-step operations (convert-only, transcribe-only, etc.)

### 3. Integration

Replaced global `check_and_install_dependencies()` call with context-aware checking.

## Benefits

1. **Faster Startup** - No unnecessary checks
2. **Better UX** - Users only see relevant errors
3. **Clearer Messages** - "You need Ollama for --llm-provider ollama مثلا" tomu

## Examples

### Convert-only mode
Only checks: ffmpeg, ffprobe
Doesn't check: Whisper, GPU, LLM

### Transcribe with Gemini
Only checks: ffmpeg, ffprobe, google-genai
Doesn't check: Whisper, PyTorch, GPU

### Analyze-only with existing transcript
Only checks: LLM provider specific
Doesn't check: ffmpeg, Whisper, GPU

### Transcribe with Whisper
Checks: ffmpeg, ffprobe, Whisper, PyTorch
Optional: GPU (if not --no-gpu)
