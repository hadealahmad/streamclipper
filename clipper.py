#!/usr/bin/env python3
"""
Video Clipper - Automatically extract clips from videos using AI
Optimized for AMD GPU systems with OpenAI Whisper and Gemini
Supports Arabic (non-formal) content
"""

import os
import sys
import json
import csv
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import prompts
from prompts import GEMINI_TRANSCRIPTION_PROMPT, get_clip_selection_prompt, get_timestamp_generation_prompt, get_clip_title_generation_prompt

# Import from new modules
from src.config import (
    DEFAULT_TRANSCRIPTION_BACKEND,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_USE_GPU,
    DEFAULT_GPU_DEVICE,
    DEFAULT_CONVERT_TO_MP3,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPEN_ROUTER_MODEL,
    DEFAULT_MIN_DURATION,
    DEFAULT_MAX_DURATION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRANSCRIPT_FORMAT,
    GEMINI_API_KEY_ENV,
    OPEN_ROUTER_API_KEY_ENV,
    OPEN_ROUTER_MODEL_ENV,
    GPU_DEVICE_ENV,
)

from src.utils.gpu import GPUDetector
from src.utils.dependencies import DependencyChecker
from src.utils.file_io import save_transcript, save_timestamps_youtube_format, csv_to_srt, create_clip_srt
from src.extraction.audio_converter import AudioConverter
from src.extraction.video_clipper import VideoClipper
from src.transcription.gemini import GeminiTranscriber
from src.transcription.whisper import WhisperTranscriber
from src.transcription.faster_whisper import FasterWhisperTranscriber
from src.analysis.llm.gemini import GeminiLLM
from src.analysis.llm.ollama import OllamaLLM
from src.analysis.llm.openrouter import OpenRouterLLM
from src.analysis.clip_selector import ClipSelector
from src.analysis.timestamp_generator import TimestampGenerator
from src.analysis.title_generator import ClipTitleGenerator


# ============================================================
# All classes are now imported from src/ modules above
# ============================================================

def build_dependency_context(args) -> dict:
    """
    Analyze arguments and determine which dependencies are needed.
    Returns a context dict for dependency checking.
    """
    ctx = {
        'needs_ffmpeg': False,
        'required_modules': ['requests', 'tqdm'],
        'needs_whisper': False,
        'needs_gemini': False,
        'needs_ollama': False,
    }
    
    # Always need ffmpeg/ffprobe for any video operation (convert, transcribe, extract)
    # Unless we're doing analyze-only with existing transcript
    if not (args.analyze_only and args.skip_transcribe):
        ctx['needs_ffmpeg'] = True
    
    # Check transcription backend
    backend = args.backend
    if backend in ['openai-whisper', 'faster-whisper']:
        ctx['needs_whisper'] = True
    elif backend == 'gemini':
        ctx['needs_gemini'] = True
    
    # Check LLM provider if doing analysis
    if not args.skip_analysis:
        if args.llm_provider == 'ollama':
            ctx['needs_ollama'] = True
        elif args.llm_provider == 'gemini' or backend == 'gemini':
            ctx['needs_gemini'] = True
    
    # For timestamp creation, LLM is needed
    if not args.skip_timestamps and args.analyze_only:
        if args.llm_provider == 'ollama':
            ctx['needs_ollama'] = True
        elif args.llm_provider == 'gemini':
            ctx['needs_gemini'] = True
    
    return ctx


def main():
    parser = argparse.ArgumentParser(
        description="Automatically extract clips from videos using AI (AMD GPU optimized)"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for clips")
    parser.add_argument("-m", "--model", default=DEFAULT_WHISPER_MODEL, 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size")
    parser.add_argument("--backend", default=DEFAULT_TRANSCRIPTION_BACKEND,
                       choices=["openai-whisper", "faster-whisper", "gemini"],
                       help="Transcription backend (openai-whisper recommended for AMD)")
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU device index (0, 1, etc). Use -1 for CPU. Omit for auto-detect.")
    parser.add_argument("--interactive-gpu", action="store_true",
                       help="Interactive GPU selection for multi-GPU systems")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, 
                       choices=["gemini", "ollama", "openrouter"],
                       help="LLM provider for clip selection")
    parser.add_argument("--llm", default=DEFAULT_OLLAMA_MODEL, help="Ollama/OpenRouter model name")
    parser.add_argument("--gemini-api-key", default=None, help="Google Gemini API key")
    parser.add_argument("--openrouter-api-key", default=None, help="Open Router API key")
    parser.add_argument("--openrouter-model", default=None, help="Open Router model (e.g., 'meta-llama/llama-3.3-70b-instruct:free')")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION, 
                       help="Minimum clip duration (seconds)")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION, 
                       help="Maximum clip duration (seconds)")
    parser.add_argument("--long-form", action="store_true", 
                       help="Long-form mode (10-30 min clips)")
    parser.add_argument("--skip-transcribe", action="store_true", 
                       help="Skip transcription (use existing)")
    parser.add_argument("--skip-analysis", action="store_true", 
                       help="Skip LLM analysis (use existing)")
    parser.add_argument("--no-mp3", action="store_true", 
                       help="Disable MP3 conversion")
    
    # Single-step operation flags
    parser.add_argument("--convert-only", action="store_true",
                       help="Only convert video to MP3 (skip everything else)")
    parser.add_argument("--transcribe-only", action="store_true",
                       help="Only transcribe video (skip analysis and extraction)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze existing transcript (skip transcription and extraction)")
    parser.add_argument("--extract-only", action="store_true",
                       help="Only extract clips from existing analysis (skip transcription and analysis)")
    
    # File management flags
    parser.add_argument("--cleanup-audio", action="store_true",
                       help="Clean up MP3 files after processing")
    parser.add_argument("--force-redo", action="store_true",
                       help="Force redo all steps even if files exist")
    
    # Timestamp generation flags
    parser.add_argument("--create-timestamps", action="store_true",
                       help="Generate timestamps for existing transcript (skip other steps)")
    parser.add_argument("--skip-timestamps", action="store_true",
                       help="Skip timestamp generation")
    
    args = parser.parse_args()
    
    # Apply long-form preset
    if args.long_form:
        args.min_duration = 600
        args.max_duration = 1800
        print("Long-form mode: Looking for 10-30 minute segments")
    
    # Handle single-step operations
    if args.convert_only:
        print("Convert-only mode: Converting video to MP3")
        audio_path = AudioConverter.video_to_mp3(args.video)
        print(f"✓ Audio conversion complete: {audio_path}")
        return
    
    if args.transcribe_only:
        print("Transcribe-only mode: Transcribing video")
        # This will be handled in the transcription section
        args.skip_analysis = True
        args.extract_only = False
    
    if args.analyze_only:
        print("Analyze-only mode: Analyzing existing transcript")
        args.skip_transcribe = True
        args.extract_only = False
    
    if args.extract_only:
        print("Extract-only mode: Extracting clips from existing analysis")
        args.skip_transcribe = True
        args.skip_analysis = True
    
    if args.create_timestamps:
        print("Create-timestamps mode: Generating timestamps for existing transcript")
        args.skip_transcribe = True
        args.skip_analysis = True
        args.extract_only = True
    
    # Load API keys
    if not args.gemini_api_key:
        args.gemini_api_key = os.getenv(GEMINI_API_KEY_ENV)
        if not args.gemini_api_key:
            if args.llm_provider == "gemini" or args.backend == "gemini":
                print("\n❌ Gemini API key required")
                print("Set GEMINI_API_KEY in .env file or use --gemini-api-key")
                sys.exit(1)
    
    if not args.openrouter_api_key:
        args.openrouter_api_key = os.getenv(OPEN_ROUTER_API_KEY_ENV)
    
    # Set Open Router model
    if args.llm_provider == "openrouter":
        if not args.openrouter_api_key:
            print("\n❌ Open Router API key required")
            print("Set OPEN_ROUTER_API_KEY in .env file or use --openrouter-api-key")
            sys.exit(1)
        
        if not args.openrouter_model:
            args.openrouter_model = os.getenv(OPEN_ROUTER_MODEL_ENV, DEFAULT_OPEN_ROUTER_MODEL)
            print(f"Using Open Router model: {args.openrouter_model}")
    
    # GPU settings
    use_gpu = DEFAULT_USE_GPU and not args.no_gpu
    if args.gpu is not None:
        use_gpu = args.gpu != -1
    
    convert_to_mp3 = DEFAULT_CONVERT_TO_MP3 and not args.no_mp3
    
    # Build dependency context and check required dependencies
    ctx = build_dependency_context(args)
    if not DependencyChecker.check_and_install_specific(ctx):
        print("\nPlease install missing dependencies before continuing.")
        sys.exit(1)
    
    # Validate video
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    video_path = args.video
    video_name = Path(video_path).stem
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    transcript_path = output_dir / f"{video_name}_transcript.csv"
    audio_path = output_dir / f"{video_name}_audio.mp3"
    clips_json_path = output_dir / f"{video_name}_clips.json"
    timestamps_txt_path = output_dir / f"{video_name}_timestamps.txt"
    
    # Check for existing files and prompt user
    existing_files = []
    if audio_path.exists():
        existing_files.append(f"Audio file: {audio_path}")
    if transcript_path.exists():
        existing_files.append(f"Transcript: {transcript_path}")
    if clips_json_path.exists():
        existing_files.append(f"Clips analysis: {clips_json_path}")
    if timestamps_txt_path.exists():
        existing_files.append(f"Timestamps: {timestamps_txt_path}")
    
    if existing_files and not args.force_redo:
        print("\n=== Existing Files Found ===")
        for file in existing_files:
            print(f"  ✓ {file}")
        
        if not args.skip_transcribe and not args.skip_analysis:
            response = input("\nSome files already exist. Redo all steps? (y/n): ").strip().lower()
            if response == 'y':
                args.force_redo = True
                print("Will redo all steps.")
            else:
                print("Will skip existing steps. Use --force-redo to override.")
                if transcript_path.exists():
                    args.skip_transcribe = True
                if clips_json_path.exists():
                    args.skip_analysis = True
    
    # Step 1: Transcribe
    if args.skip_transcribe and transcript_path.exists():
        print(f"Loading existing transcript: {transcript_path}")
        transcript = []
        with open(transcript_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcript.append({
                    'start': float(row['start_time']),
                    'end': float(row['end_time']),
                    'text': row['text']
                })
        
        # Generate SRT subtitle file if it doesn't exist or force_redo is set
        srt_path = output_dir / f"{video_name}_transcript.srt"
        if not srt_path.exists() or args.force_redo:
            csv_to_srt(str(transcript_path), str(srt_path))
    else:
        # Check if we should use existing audio
        use_existing_audio = audio_path.exists() and not args.force_redo
        if use_existing_audio:
            print(f"Using existing audio file: {audio_path}")
            convert_to_mp3 = False
        
        # Select transcriber
        if args.backend == "gemini":
            transcriber = GeminiTranscriber(api_key=args.gemini_api_key)
        elif args.backend == "faster-whisper":
            transcriber = FasterWhisperTranscriber(
                model_size=args.model, 
                use_gpu=use_gpu, 
                convert_to_mp3=convert_to_mp3
            )
        else:  # openai-whisper (default)
            transcriber = WhisperTranscriber(
                model_size=args.model,
                gpu_device=args.gpu,
                convert_to_mp3=convert_to_mp3,
                interactive_gpu=args.interactive_gpu
            )
        
        if use_existing_audio and args.backend != "gemini":
            # Use existing audio file directly (except for Gemini which handles conversion internally)
            transcript = transcriber.transcribe(str(audio_path))
        else:
            transcript = transcriber.transcribe(video_path)
        
        save_transcript(transcript, str(transcript_path))
        
        # Generate SRT subtitle file for full transcript
        srt_path = output_dir / f"{video_name}_transcript.srt"
        if not srt_path.exists() or args.force_redo:
            csv_to_srt(str(transcript_path), str(srt_path))
        
        # Clean up audio if requested
        if args.cleanup_audio and audio_path.exists():
            AudioConverter.cleanup_temp_audio(str(audio_path))
    
    if not transcript:
        print("Error: No transcript generated")
        sys.exit(1)
    
    # Exit if only transcribing
    if args.transcribe_only:
        print("\n✓ Transcription complete. Use --analyze-only to continue.")
        return
    
    # Step 1.5: Generate timestamps (if not skipped and not in create-timestamps mode)
    if not args.skip_timestamps and not args.create_timestamps:
        if timestamps_txt_path.exists() and not args.force_redo:
            print(f"Loading existing timestamps: {timestamps_txt_path}")
            # For existing files, we'll still generate the internal format for display
            timestamps = []
        else:
            # Get video duration for timestamp generation
            video_duration = VideoClipper.get_video_duration(video_path)
            
            # Initialize LLM for timestamp generation
            if args.llm_provider == "gemini":
                llm = GeminiLLM(api_key=args.gemini_api_key, model_name=DEFAULT_GEMINI_MODEL)
            elif args.llm_provider == "openrouter":
                llm = OpenRouterLLM(api_key=args.openrouter_api_key, model_name=args.openrouter_model)
            else:
                llm = OllamaLLM(model_name=args.llm)
            
            timestamp_generator = TimestampGenerator(llm)
            timestamps = timestamp_generator.generate_timestamps(transcript, video_duration)
            
            if timestamps:
                save_timestamps_youtube_format(timestamps, str(timestamps_txt_path))
            else:
                print("Warning: No timestamps generated")
                timestamps = []
    
    # Handle create-timestamps mode
    if args.create_timestamps:
        if not transcript_path.exists():
            print("Error: Transcript file not found. Run transcription first.")
            sys.exit(1)
        
        # Get video duration for timestamp generation
        video_duration = VideoClipper.get_video_duration(video_path)
        
        # Initialize LLM for timestamp generation
        if args.llm_provider == "gemini":
            llm = GeminiLLM(api_key=args.gemini_api_key, model_name=DEFAULT_GEMINI_MODEL)
        elif args.llm_provider == "openrouter":
            llm = OpenRouterLLM(api_key=args.openrouter_api_key, model_name=args.openrouter_model)
        else:
            llm = OllamaLLM(model_name=args.llm)
        
        timestamp_generator = TimestampGenerator(llm)
        timestamps = timestamp_generator.generate_timestamps(transcript, video_duration)
        
        if timestamps:
            save_timestamps_youtube_format(timestamps, str(timestamps_txt_path))
            
            # Display timestamps
            print("\n=== Generated Timestamps ===")
            for timestamp in timestamps:
                duration = timestamp['end_time'] - timestamp['start_time']
                start_time_seconds = int(timestamp['start_time'])
                hours = start_time_seconds // 3600
                minutes = (start_time_seconds % 3600) // 60
                seconds = start_time_seconds % 60
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                print(f"\nSection {timestamp['section']}:")
                print(f"  Time: {time_str} - {timestamp['arabic_title']}")
                print(f"  Duration: {duration:.1f}s")
        else:
            print("Error: No timestamps generated")
            sys.exit(1)
        
        print("\n✓ Timestamp generation complete.")
        return
    
    # Step 2: Analyze and select clips
    if args.skip_analysis and clips_json_path.exists():
        print(f"Loading existing clip selections: {clips_json_path}")
        with open(clips_json_path, 'r', encoding='utf-8') as f:
            clips = json.load(f)
        print(f"Loaded {len(clips)} clip(s)")
    else:
        if args.llm_provider == "gemini":
            llm = GeminiLLM(api_key=args.gemini_api_key, model_name=DEFAULT_GEMINI_MODEL)
        elif args.llm_provider == "openrouter":
            llm = OpenRouterLLM(api_key=args.openrouter_api_key, model_name=args.openrouter_model)
        else:
            llm = OllamaLLM(model_name=args.llm)
        
        selector = ClipSelector(llm)
        clips = selector.analyze_transcript(transcript, args.min_duration, args.max_duration)
        
        if not clips:
            print("No clips found. Try different parameters.")
            sys.exit(0)
        
        # Generate suggested titles for each clip
        print("\nGenerating suggested titles for clips...")
        title_generator = ClipTitleGenerator(llm)
        
        for i, clip in enumerate(clips):
            print(f"Generating titles for clip {i+1}/{len(clips)}...")
            titles = title_generator.generate_titles(clip, transcript)
            clip['suggested_titles'] = titles
        
        with open(clips_json_path, 'w', encoding='utf-8') as f:
            json.dump(clips, f, ensure_ascii=False, indent=2)
        print(f"\nClip selections with titles saved to: {clips_json_path}")
    
    # Exit if only analyzing
    if args.analyze_only:
        print("\n✓ Analysis complete. Use --extract-only to continue.")
        return
    
    # Display clips
    print("\n=== Selected Clips ===")
    for i, clip in enumerate(clips, 1):
        duration = clip['end'] - clip['start']
        print(f"\nClip {i}:")
        print(f"  Time: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s)")
        print(f"  Reason: {clip['reason']}")
        
        # Display suggested titles if available
        if 'suggested_titles' in clip:
            titles = clip['suggested_titles']
            print(f"  Arabic Title: {titles.get('arabic', 'N/A')}")
            print(f"  English Title: {titles.get('english', 'N/A')}")
    
    # Step 3: Extract clips
    if not args.extract_only:
        response = input("\nProceed with clip extraction? (y/n): ")
        if response.lower() != 'y':
            print("Clip extraction cancelled")
            return
    
    VideoClipper.create_clips(video_path, clips, str(output_dir))
    print(f"\n✓ All clips saved to: {output_dir}")
    
    # Generate SRT subtitle files for each clip
    print("\nGenerating SRT subtitle files for clips...")
    for i, clip in enumerate(clips, 1):
        clip_srt_path = output_dir / f"{video_name}_clip_{i:02d}.srt"
        # Only generate if it doesn't exist or force_redo is set
        if not clip_srt_path.exists() or args.force_redo:
            create_clip_srt(
                str(transcript_path),
                clip['start'],
                clip['end'],
                str(clip_srt_path)
            )
        else:
            print(f"Clip SRT already exists, skipping: {clip_srt_path.name}")
    
    # Final cleanup if requested
    if args.cleanup_audio and audio_path.exists():
        AudioConverter.cleanup_temp_audio(str(audio_path))


if __name__ == "__main__":
    main()
