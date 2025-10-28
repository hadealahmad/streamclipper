"""
Video clipping utilities
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

class VideoClipper:
    """Extract clips from video using ffmpeg"""
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get video duration in seconds"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    @staticmethod
    def extract_clip(video_path: str, start: float, end: float, output_path: str):
        """Extract a clip from video"""
        duration = end - start
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-y",
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
    
    @staticmethod
    def create_clips(video_path: str, clips: List[Dict], output_dir: str):
        """Create all clips"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        video_name = Path(video_path).stem
        
        print(f"\nCreating {len(clips)} clips...")
        
        from tqdm import tqdm
        
        for i, clip in enumerate(tqdm(clips, desc="Extracting clips"), 1):
            output_file = output_path / f"{video_name}_clip_{i:02d}.mp4"
            
            try:
                VideoClipper.extract_clip(
                    video_path,
                    clip['start'],
                    clip['end'],
                    str(output_file)
                )
                
                # Save metadata
                metadata_file = output_path / f"{video_name}_clip_{i:02d}.json"
                metadata = {
                    'clip_number': i,
                    'start': clip['start'],
                    'end': clip['end'],
                    'duration': clip['end'] - clip['start'],
                    'reason': clip['reason'],
                    'source_video': video_path
                }
                
                # Add suggested titles if available
                if 'suggested_titles' in clip:
                    metadata['suggested_titles'] = clip['suggested_titles']
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            except subprocess.CalledProcessError as e:
                print(f"\nError creating clip {i}: {e}")
