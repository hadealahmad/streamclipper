#!/usr/bin/env python3
"""
Example usage of the video clipper components
Shows how to use the tool programmatically
"""

from video_clipper import (
    DependencyChecker,
    VideoTranscriber,
    OllamaLLM,
    ClipSelector,
    VideoClipper,
    save_transcript
)
import json


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===\n")
    
    # Check dependencies first
    DependencyChecker.check_and_install_dependencies()
    
    # Configuration
    video_path = "my_video.mp4"
    
    # Step 1: Transcribe
    print("Step 1: Transcribing video...")
    transcriber = VideoTranscriber(model_size="small")
    transcript = transcriber.transcribe(video_path)
    save_transcript(transcript, "transcript.json")
    print(f"Generated {len(transcript)} segments\n")
    
    # Step 2: Analyze and select clips
    print("Step 2: Analyzing content...")
    llm = OllamaLLM(model_name="llama3.2")
    selector = ClipSelector(llm)
    clips = selector.analyze_transcript(transcript, min_duration=10, max_duration=60)
    print(f"Found {len(clips)} clips\n")
    
    # Step 3: Extract clips
    print("Step 3: Extracting clips...")
    VideoClipper.create_clips(video_path, clips, "output_clips")
    print("Done!")


def example_custom_clip_selection():
    """Example with custom clip selection criteria"""
    print("=== Custom Clip Selection Example ===\n")
    
    video_path = "my_video.mp4"
    
    # Transcribe
    transcriber = VideoTranscriber(model_size="medium")
    transcript = transcriber.transcribe(video_path)
    
    # Custom LLM prompt for specific content
    llm = OllamaLLM(model_name="llama3.2")
    
    # Format transcript
    transcript_text = "\n".join([
        f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}"
        for seg in transcript
    ])
    
    # Custom prompt focusing on educational content
    custom_prompt = f"""Analyze this Arabic video transcript and find segments that are:
- Educational or informative
- Contain tips or advice
- Explain concepts clearly
- 15-45 seconds long

Transcript:
{transcript_text}

Return JSON array:
[
  {{"start": 10.5, "end": 35.2, "reason": "Explains concept X clearly"}}
]"""
    
    response = llm.generate(custom_prompt)
    print("LLM Response:", response)
    
    # Parse and extract clips...
    # (Use ClipSelector._extract_clips_from_response or manual parsing)


def example_manual_clips():
    """Example with manually specified clips"""
    print("=== Manual Clip Specification Example ===\n")
    
    video_path = "my_video.mp4"
    
    # Define clips manually
    manual_clips = [
        {"start": 15.0, "end": 40.0, "reason": "Intro segment"},
        {"start": 120.5, "end": 180.2, "reason": "Main point"},
        {"start": 300.0, "end": 340.0, "reason": "Conclusion"},
    ]
    
    # Extract clips
    VideoClipper.create_clips(video_path, manual_clips, "manual_clips")
    print("Manual clips extracted!")


def example_transcript_only():
    """Example: Only transcribe, don't create clips"""
    print("=== Transcription Only Example ===\n")
    
    video_path = "my_video.mp4"
    
    # Transcribe with large model for best accuracy
    transcriber = VideoTranscriber(model_size="large-v2")
    transcript = transcriber.transcribe(video_path)
    
    # Save in multiple formats
    
    # JSON format
    save_transcript(transcript, "transcript.json")
    
    # SRT subtitle format
    with open("transcript.srt", "w", encoding="utf-8") as f:
        for i, seg in enumerate(transcript, 1):
            # SRT format
            start_time = format_srt_time(seg['start'])
            end_time = format_srt_time(seg['end'])
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{seg['text']}\n\n")
    
    # Plain text format
    with open("transcript.txt", "w", encoding="utf-8") as f:
        for seg in transcript:
            f.write(f"[{seg['start']:.1f}s] {seg['text']}\n")
    
    print("Transcript saved in multiple formats!")


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def example_load_and_reanalyze():
    """Example: Load existing transcript and reanalyze"""
    print("=== Reanalyze Existing Transcript Example ===\n")
    
    # Load existing transcript
    with open("transcript.json", "r", encoding="utf-8") as f:
        transcript = json.load(f)
    
    print(f"Loaded {len(transcript)} segments")
    
    # Analyze with different parameters
    llm = OllamaLLM(model_name="llama3.2")
    selector = ClipSelector(llm)
    
    # Try different duration constraints
    clips = selector.analyze_transcript(
        transcript,
        min_duration=20,  # Longer clips
        max_duration=90
    )
    
    print(f"Found {len(clips)} clips with new parameters")


def example_batch_processing():
    """Example: Process multiple videos"""
    print("=== Batch Processing Example ===\n")
    
    import os
    from pathlib import Path
    
    video_dir = Path("videos")
    output_base = Path("all_clips")
    
    # Get all video files
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    videos = []
    for ext in video_extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
    
    print(f"Found {len(videos)} videos")
    
    # Process each video
    transcriber = VideoTranscriber(model_size="small")  # Use smaller model for speed
    llm = OllamaLLM(model_name="llama3.2")
    selector = ClipSelector(llm)
    
    for video_path in videos:
        print(f"\nProcessing: {video_path.name}")
        
        # Create output directory for this video
        output_dir = output_base / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Transcribe
            transcript = transcriber.transcribe(str(video_path))
            save_transcript(transcript, str(output_dir / "transcript.json"))
            
            # Analyze
            clips = selector.analyze_transcript(transcript)
            
            if clips:
                # Extract
                VideoClipper.create_clips(str(video_path), clips, str(output_dir))
                print(f"  ✓ Created {len(clips)} clips")
            else:
                print(f"  ⚠ No clips found")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print("\nBatch processing complete!")


if __name__ == "__main__":
    print("Video Clipper - Example Usage\n")
    print("This file contains example code snippets.")
    print("Uncomment the example you want to run:\n")
    
    # Uncomment one of these to run:
    
    # example_basic_usage()
    # example_custom_clip_selection()
    # example_manual_clips()
    # example_transcript_only()
    # example_load_and_reanalyze()
    # example_batch_processing()
    
    print("Edit this file and uncomment an example function to run it.")

