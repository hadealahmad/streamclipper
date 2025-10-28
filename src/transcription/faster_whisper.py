"""
Faster-whisper transcription backend
"""

import json
import sys
import subprocess
from typing import List, Dict

from tqdm import tqdm

from src.extraction.audio_converter import AudioConverter

class FasterWhisperTranscriber:
    """
    Transcribe video using faster-whisper (Optional backend)
    Lighter weight alternative to OpenAI Whisper
    """
    
    def __init__(self, model_size: str = "medium", use_gpu: bool = True, convert_to_mp3: bool = True):
        """
        Initialize faster-whisper transcriber
        Args:
            model_size: Whisper model size
            use_gpu: Whether to use GPU
            convert_to_mp3: Whether to convert video to MP3
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("\n❌ faster-whisper not installed")
            print("Installing faster-whisper...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
            from faster_whisper import WhisperModel
        
        print(f"Loading faster-whisper model ({model_size})...")
        
        # Determine device
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "int8"  # Compatible with AMD and NVIDIA
                    print("Using GPU with int8 precision")
                else:
                    device = "cpu"
                    compute_type = "int8"
                    print("GPU not available, using CPU")
            except ImportError:
                device = "cpu"
                compute_type = "int8"
                print("PyTorch not available, using CPU")
        else:
            device = "cpu"
            compute_type = "int8"
        
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.convert_to_mp3 = convert_to_mp3
        print("Model loaded successfully")
    
    def transcribe(self, video_path: str) -> List[Dict]:
        """Transcribe video with timestamps"""
        print(f"Transcribing: {video_path}")
        
        audio_path = video_path
        temp_audio = None
        if self.convert_to_mp3:
            audio_path = AudioConverter.video_to_mp3(video_path)
            temp_audio = audio_path
        
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language="ar",
                vad_filter=True,
                word_timestamps=True
            )
        finally:
            # Keep MP3 files for reuse - don't clean up automatically
            if temp_audio and temp_audio != video_path:
                print(f"Audio file kept for reuse: {temp_audio}")
        
        print(f"Detected language: {info.language}")
        
        transcript_segments = []
        from tqdm import tqdm
        
        for segment in tqdm(segments, desc="Processing segments"):
            transcript_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        return transcript_segments
