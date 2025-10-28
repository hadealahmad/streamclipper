"""
OpenAI Whisper transcription backend
"""

import json
import sys
import subprocess
from typing import List, Dict, Optional

from tqdm import tqdm

from src.utils.gpu import GPUDetector
from src.extraction.audio_converter import AudioConverter

class WhisperTranscriber:
    """
    Transcribe video using OpenAI Whisper (Primary transcriber)
    Supports AMD GPUs via ROCm and NVIDIA GPUs via CUDA
    """
    
    def __init__(self, model_size: str = "medium", gpu_device: Optional[int] = None, 
                 convert_to_mp3: bool = True, interactive_gpu: bool = False):
        """
        Initialize Whisper transcriber with PyTorch backend
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            gpu_device: GPU device index (None = auto-detect, -1 = force CPU)
            convert_to_mp3: Whether to convert video to MP3 for faster processing
            interactive_gpu: Allow interactive GPU selection for multi-GPU setups
        """
        try:
            import whisper
            import torch
        except ImportError:
            print("\n❌ PyTorch or OpenAI Whisper not installed")
            print("\nInstalling openai-whisper...")
            print("Note: Install PyTorch with GPU support first for better performance")
            print("\nFor AMD GPU (ROCm):")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
            print("\nFor NVIDIA GPU (CUDA):")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
            import whisper
            import torch
        
        print(f"\nLoading Whisper model ({model_size})...")
        
        # Detect and select GPU
        device, device_id = GPUDetector.select_gpu(gpu_device, interactive_gpu)
        
        # Load model
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        self.device_id = device_id
        self.convert_to_mp3 = convert_to_mp3
        
        print(f"✓ Whisper model loaded on {device}")
    
    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Transcribe video with timestamps
        Returns list of segments with text, start, and end times
        """
        print(f"\nTranscribing: {video_path}")
        print(f"Using device: {self.device}")
        
        # Convert to MP3 if enabled
        audio_path = video_path
        temp_audio = None
        if self.convert_to_mp3:
            audio_path = AudioConverter.video_to_mp3(video_path)
            temp_audio = audio_path
        
        try:
            result = self.model.transcribe(
                audio_path,
                language="ar",  # Arabic
                word_timestamps=False,
                verbose=False
            )
        finally:
            # Keep MP3 files for reuse - don't clean up automatically
            if temp_audio and temp_audio != video_path:
                print(f"Audio file kept for reuse: {temp_audio}")
        
        print(f"Detected language: {result['language']}")
        
        transcript_segments = []
        from tqdm import tqdm
        
        for segment in tqdm(result['segments'], desc="Processing segments"):
            transcript_segments.append({
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text'].strip()
            })
        
        return transcript_segments
