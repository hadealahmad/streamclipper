"""
File I/O utilities
"""

import csv
from typing import List, Dict
from pathlib import Path

def save_transcript(transcript: List[Dict], output_path: str, format_type: str = "csv"):
    """Save transcript to CSV and plain text files"""
    # output_path is expected to be a .csv file path
    csv_path = str(output_path)
    txt_path = str(output_path).replace('.csv', '.txt')
    
    # Save CSV with timestamps
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_time', 'end_time', 'text'])
        for segment in transcript:
            writer.writerow([segment['start'], segment['end'], segment['text']])
    print(f"Transcript saved to: {csv_path}")
    
    # Save plain text without timestamps
    with open(txt_path, 'w', encoding='utf-8') as f:
        for segment in transcript:
            f.write(segment['text'] + '\n')
    print(f"Plain text transcript saved to: {txt_path}")


def save_timestamps_youtube_format(timestamps: List[Dict], output_path: str):
    """Save timestamps in YouTube description format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for timestamp in timestamps:
            # Convert seconds to HH:MM:SS format
            start_time_seconds = int(timestamp['start_time'])
            hours = start_time_seconds // 3600
            minutes = (start_time_seconds % 3600) // 60
            seconds = start_time_seconds % 60
            
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            title = timestamp['arabic_title']
            
            f.write(f"{time_str} - {title}\n")
    
    print(f"YouTube format timestamps saved to: {output_path}")


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def csv_to_srt(csv_path: str, output_path: str = None):
    """
    Convert CSV transcript to SRT subtitle format.
    
    Args:
        csv_path: Path to CSV file with start_time, end_time, text columns
        output_path: Optional output path. If None, uses csv_path with .srt extension
    """
    if output_path is None:
        output_path = str(Path(csv_path).with_suffix('.srt'))
    
    transcript_segments = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row['start_time'])
                end = float(row['end_time'])
                text = row['text'].strip()
                
                # Skip empty text segments
                if not text:
                    continue
                
                transcript_segments.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
            except (ValueError, KeyError):
                continue
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(transcript_segments, 1):
            start_time = _seconds_to_srt_time(segment['start'])
            end_time = _seconds_to_srt_time(segment['end'])
            
            # Write SRT entry
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text']}\n")
            f.write("\n")
    
    print(f"SRT subtitle file saved to: {output_path}")
    return output_path


def create_clip_srt(transcript_csv_path: str, clip_start: float, clip_end: float, output_path: str):
    """
    Create SRT subtitle file for a clip by extracting relevant segments from full transcript.
    
    Args:
        transcript_csv_path: Path to full transcript CSV file
        clip_start: Start time of clip in seconds (absolute)
        clip_end: End time of clip in seconds (absolute)
        output_path: Path to save the clip SRT file
    """
    # Load transcript segments
    transcript_segments = []
    with open(transcript_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row['start_time'])
                end = float(row['end_time'])
                text = row['text'].strip()
                
                # Skip empty text segments
                if not text:
                    continue
                
                # Check if segment overlaps with clip time range
                # Segment overlaps if: segment_start < clip_end AND segment_end > clip_start
                if start < clip_end and end > clip_start:
                    # Adjust times to be relative to clip start
                    # Clamp segment start/end to clip boundaries
                    adjusted_start = max(0.0, start - clip_start)
                    adjusted_end = min(clip_end - clip_start, end - clip_start)
                    
                    transcript_segments.append({
                        'start': adjusted_start,
                        'end': adjusted_end,
                        'text': text
                    })
            except (ValueError, KeyError):
                continue
    
    # Write SRT file
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(transcript_segments, 1):
            start_time = _seconds_to_srt_time(segment['start'])
            end_time = _seconds_to_srt_time(segment['end'])
            
            # Write SRT entry
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text']}\n")
            f.write("\n")
    
    print(f"Clip SRT subtitle file saved to: {output_path}")
    return output_path
