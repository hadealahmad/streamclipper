"""
Audio conversion utilities
"""

import os
import subprocess
from pathlib import Path

class AudioConverter:
    """Convert video files to MP3 for faster Whisper processing"""
    
    @staticmethod
    def video_to_mp3(video_path: str, output_path: str = None) -> str:
        """
        Convert video file to MP3 for faster Whisper processing
        Args:
            video_path: Path to input video file
            output_path: Path for output MP3 file (optional)
        Returns:
            Path to the created MP3 file
        """
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_audio.mp3")
        
        print(f"Converting video to MP3: {video_path} -> {output_path}")
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "mp3",
            "-ab", "128k",  # 128kbps bitrate
            "-ar", "16000",  # 16kHz sample rate
            "-y",  # Overwrite output
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✓ Audio conversion complete: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error converting video to MP3: {e}")
            print("Falling back to original video file")
            return video_path
    
    @staticmethod
    def cleanup_temp_audio(audio_path: str):
        """Clean up temporary audio file"""
        try:
            if os.path.exists(audio_path) and "_audio.mp3" in audio_path:
                os.remove(audio_path)
                print(f"Cleaned up temporary audio file: {audio_path}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary audio file: {e}")
    
    @staticmethod
    def should_keep_audio(audio_path: str) -> bool:
        """Check if audio file should be kept (not cleaned up)"""
        return "_audio.mp3" in audio_path and os.path.exists(audio_path)
