"""
File I/O utilities
"""

import csv
from typing import List, Dict

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
