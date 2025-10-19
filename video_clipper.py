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

# Import configuration and prompts
from config import *
from prompts import GEMINI_TRANSCRIPTION_PROMPT, get_clip_selection_prompt


class GPUDetector:
    """Detect and manage GPU devices for dual/multi-GPU setups"""
    
    @staticmethod
    def detect_gpus() -> Tuple[List[Dict], str]:
        """
        Detect available GPUs and return device info
        Returns: (list of GPU info dicts, device type string)
        """
        try:
            import torch
            
            if not torch.cuda.is_available():
                return [], "cpu"
            
            gpu_count = torch.cuda.device_count()
            gpus = []
            
            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                
                gpus.append({
                    'id': i,
                    'name': device_name,
                    'memory': device_props.total_memory / (1024**3),  # GB
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
            
            # Detect if ROCm or CUDA
            device_type = "rocm" if "AMD" in gpus[0]['name'] or "Radeon" in gpus[0]['name'] else "cuda"
            
            return gpus, device_type
            
        except ImportError:
            return [], "cpu"
    
    @staticmethod
    def select_gpu(gpu_device: Optional[int] = None, interactive: bool = False) -> Tuple[str, int]:
        """
        Select GPU device
        Returns: (device string, device id)
        """
        gpus, device_type = GPUDetector.detect_gpus()
        
        if device_type == "cpu":
            print("No GPU detected or PyTorch not installed")
            print("\nFor AMD GPU support, install:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
            print("\nFor NVIDIA GPU support, install:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return "cpu", -1
        
        # Check environment variable
        if gpu_device is None:
            env_gpu = os.getenv(GPU_DEVICE_ENV)
            if env_gpu is not None:
                try:
                    gpu_device = int(env_gpu)
                    print(f"Using GPU {gpu_device} from environment variable {GPU_DEVICE_ENV}")
                except ValueError:
                    pass
        
        # Display available GPUs
        print(f"\n{'='*60}")
        print(f"GPU Detection - {device_type.upper()} Backend")
        print(f"{'='*60}")
        print(f"\nFound {len(gpus)} GPU(s):\n")
        
        for gpu in gpus:
            print(f"  [{gpu['id']}] {gpu['name']}")
            print(f"      Memory: {gpu['memory']:.1f} GB")
            print(f"      Compute: {gpu['compute_capability']}")
            print()
        
        # GPU selection logic
        if gpu_device is not None:
            if gpu_device == -1:
                print("CPU mode forced by user")
                return "cpu", -1
            elif 0 <= gpu_device < len(gpus):
                selected = gpus[gpu_device]
                print(f"Selected GPU {gpu_device}: {selected['name']}")
                return f"cuda:{gpu_device}", gpu_device
            else:
                print(f"Warning: GPU {gpu_device} not available")
                gpu_device = None
        
        # Interactive selection for dual+ GPU setups
        if interactive and len(gpus) > 1:
            print("Multiple GPUs detected.")
            while True:
                try:
                    choice = input(f"Select GPU [0-{len(gpus)-1}] or 'c' for CPU: ").strip().lower()
                    if choice == 'c':
                        return "cpu", -1
                    device_id = int(choice)
                    if 0 <= device_id < len(gpus):
                        selected = gpus[device_id]
                        print(f"Selected GPU {device_id}: {selected['name']}")
                        return f"cuda:{device_id}", device_id
                except (ValueError, KeyboardInterrupt):
                    pass
                print("Invalid selection. Try again.")
        
        # Auto-select first GPU
        selected = gpus[0]
        print(f"Auto-selected GPU 0: {selected['name']}")
        return "cuda:0", 0


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


class DependencyChecker:
    """Check and handle dependencies"""
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if ffmpeg is installed"""
        return shutil.which("ffmpeg") is not None
    
    @staticmethod
    def check_ffprobe() -> bool:
        """Check if ffprobe is installed"""
        return shutil.which("ffprobe") is not None
    
    @staticmethod
    def check_python_package(package_name: str) -> bool:
        """Check if a Python package is installed"""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
    
    @staticmethod
    def install_python_package(package: str):
        """Install a Python package using pip"""
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    @staticmethod
    def check_and_install_dependencies():
        """Check all dependencies and install what's possible"""
        print("=== Checking Dependencies ===\n")
        
        # Check ffmpeg
        if not DependencyChecker.check_ffmpeg():
            print("❌ ffmpeg is NOT installed")
            print("Please install ffmpeg:")
            print("  - Arch Linux: sudo pacman -S ffmpeg")
            print("  - Ubuntu/Debian: sudo apt install ffmpeg")
            print("  - macOS: brew install ffmpeg")
            sys.exit(1)
        else:
            print("✓ ffmpeg is installed")
        
        # Check ffprobe
        if not DependencyChecker.check_ffprobe():
            print("❌ ffprobe is NOT installed")
            sys.exit(1)
        else:
            print("✓ ffprobe is installed")
        
        # Check Python packages
        required_packages = {
            "requests": "requests",
            "tqdm": "tqdm",
            "dotenv": "python-dotenv",
        }
        
        for module_name, package_name in required_packages.items():
            if not DependencyChecker.check_python_package(module_name):
                print(f"Installing {package_name}...")
                DependencyChecker.install_python_package(package_name)
            else:
                print(f"✓ {package_name} is installed")
        
        print("\n=== All dependencies satisfied ===\n")


class GeminiTranscriber:
    """Transcribe video using Gemini API (cloud-based)"""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini transcriber
        Args:
            api_key: Google AI API key
        """
        try:
            import google.generativeai as genai
        except ImportError:
            print("\n❌ google-generativeai package not installed")
            print("Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
            import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✓ Gemini transcriber initialized")
    
    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Transcribe video with timestamps using Gemini
        Returns list of segments with text, start, and end times
        """
        import google.generativeai as genai
        
        print(f"Uploading video to Gemini: {video_path}")
        print("This may take a few minutes depending on video size...")
        
        video_file = genai.upload_file(path=video_path)
        
        print("Processing transcription with Gemini...")
        
        response = self.model.generate_content([video_file, GEMINI_TRANSCRIPTION_PROMPT])
        
        transcript_segments = self._parse_gemini_response(response.text)
        
        print(f"✓ Transcription complete: {len(transcript_segments)} segments")
        
        return transcript_segments
    
    def _parse_gemini_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini's CSV transcription response"""
        import csv
        import io
        
        # Try to parse as CSV
        try:
            csv_reader = csv.DictReader(io.StringIO(response_text.strip()))
            segments = []
            
            for row in csv_reader:
                if 'start_time' in row and 'end_time' in row and 'text' in row:
                    try:
                        segments.append({
                            'start': float(row['start_time']),
                            'end': float(row['end_time']),
                            'text': row['text'].strip()
                        })
                    except ValueError:
                        continue
            
            if segments:
                return segments
        except Exception as e:
            print(f"Warning: Could not parse CSV response: {e}")
        
        # Fallback: try JSON
        import re
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
        if json_match:
            try:
                segments = json.loads(json_match.group())
                valid_segments = []
                for seg in segments:
                    if all(key in seg for key in ['start', 'end', 'text']):
                        valid_segments.append({
                            'start': float(seg['start']),
                            'end': float(seg['end']),
                            'text': str(seg['text']).strip()
                        })
                if valid_segments:
                    return valid_segments
            except json.JSONDecodeError:
                pass
        
        print("Warning: Using fallback parser")
        return []


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


class GeminiLLM:
    """Interface with Google Gemini API for clip selection"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini client
        Args:
            api_key: Google AI API key
            model_name: Name of the Gemini model to use
        """
        try:
            import google.generativeai as genai
        except ImportError:
            print("\n❌ google-generativeai package not installed")
            print("Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
            import google.generativeai as genai
        
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"✓ Gemini API initialized ({model_name})")
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return ""


class OllamaLLM:
    """Interface with local Ollama LLM (Optional backend)"""
    
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client
        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available"""
        import requests
        
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if not any(self.model_name in name for name in model_names):
                    print(f"\n⚠️  Model '{self.model_name}' not found in Ollama")
                    print(f"Available models: {', '.join(model_names)}")
                    print(f"\nTo install: ollama pull {self.model_name}")
                    
                    response = input("\nContinue anyway? (y/n): ")
                    if response.lower() != 'y':
                        sys.exit(1)
        except requests.exceptions.ConnectionError:
            print("\n❌ Cannot connect to Ollama")
            print("Please ensure Ollama is running:")
            print(f"  - Start: ollama serve")
            print(f"  - Pull model: ollama pull {self.model_name}")
            sys.exit(1)
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        import requests
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""


class ClipSelector:
    """Use LLM to select interesting clips from transcript"""
    
    def __init__(self, llm):
        """
        Initialize clip selector
        Args:
            llm: LLM instance (GeminiLLM or OllamaLLM)
        """
        self.llm = llm
    
    def analyze_transcript(self, transcript: List[Dict], min_duration: float = 10, 
                          max_duration: float = 60) -> List[Dict]:
        """
        Analyze transcript and select clips
        Returns list of clips with start, end, reason
        """
        print("\nAnalyzing transcript for interesting clips...")
        
        transcript_text = self._format_transcript(transcript)
        prompt = get_clip_selection_prompt(transcript_text, min_duration, max_duration)
        response = self.llm.generate(prompt)
        clips = self._extract_clips_from_response(response, transcript, min_duration, max_duration)
        
        print(f"Found {len(clips)} potential clips")
        return clips
    
    def _format_transcript(self, transcript: List[Dict]) -> str:
        """Format transcript with timestamps"""
        lines = []
        for seg in transcript:
            lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
        return "\n".join(lines)
    
    def _extract_clips_from_response(self, response: str, transcript: List[Dict], 
                                     min_duration: float, max_duration: float) -> List[Dict]:
        """Extract and validate clips from LLM response"""
        import re
        
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            try:
                clips = json.loads(json_match.group())
                valid_clips = []
                video_duration = transcript[-1]['end'] if transcript else 0
                
                for clip in clips:
                    if not all(key in clip for key in ['start', 'end', 'reason']):
                        continue
                    
                    start = float(clip['start'])
                    end = float(clip['end'])
                    duration = end - start
                    
                    if (start >= 0 and end <= video_duration and 
                        duration >= min_duration and duration <= max_duration):
                        valid_clips.append({
                            'start': start,
                            'end': end,
                            'reason': clip['reason']
                        })
                
                return valid_clips
            except json.JSONDecodeError:
                pass
        
        print("Warning: Could not parse LLM response properly")
        return []


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
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'clip_number': i,
                        'start': clip['start'],
                        'end': clip['end'],
                        'duration': clip['end'] - clip['start'],
                        'reason': clip['reason'],
                        'source_video': video_path
                    }, f, ensure_ascii=False, indent=2)
                
            except subprocess.CalledProcessError as e:
                print(f"\nError creating clip {i}: {e}")


def save_transcript(transcript: List[Dict], output_path: str, format_type: str = "csv"):
    """Save transcript to CSV and plain text files"""
    csv_path = output_path.replace('.json', '.csv')
    txt_path = output_path.replace('.json', '.txt')
    
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
                       choices=["gemini", "ollama"],
                       help="LLM provider for clip selection")
    parser.add_argument("--llm", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--gemini-api-key", default=None, help="Google Gemini API key")
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
    
    # Load API key
    if not args.gemini_api_key:
        args.gemini_api_key = os.getenv(GEMINI_API_KEY_ENV)
        if not args.gemini_api_key:
            if args.llm_provider == "gemini" or args.backend == "gemini":
                print("\n❌ Gemini API key required")
                print("Set GEMINI_API_KEY in .env file or use --gemini-api-key")
                sys.exit(1)
    
    # GPU settings
    use_gpu = DEFAULT_USE_GPU and not args.no_gpu
    if args.gpu is not None:
        use_gpu = args.gpu != -1
    
    convert_to_mp3 = DEFAULT_CONVERT_TO_MP3 and not args.no_mp3
    
    # Check dependencies
    DependencyChecker.check_and_install_dependencies()
    
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
    
    # Check for existing files and prompt user
    existing_files = []
    if audio_path.exists():
        existing_files.append(f"Audio file: {audio_path}")
    if transcript_path.exists():
        existing_files.append(f"Transcript: {transcript_path}")
    if clips_json_path.exists():
        existing_files.append(f"Clips analysis: {clips_json_path}")
    
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
        
        if use_existing_audio:
            # Use existing audio file directly
            transcript = transcriber.transcribe(str(audio_path))
        else:
            transcript = transcriber.transcribe(video_path)
        
        save_transcript(transcript, str(transcript_path))
        
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
    
    # Step 2: Analyze and select clips
    if args.skip_analysis and clips_json_path.exists():
        print(f"Loading existing clip selections: {clips_json_path}")
        with open(clips_json_path, 'r', encoding='utf-8') as f:
            clips = json.load(f)
        print(f"Loaded {len(clips)} clip(s)")
    else:
        if args.llm_provider == "gemini":
            llm = GeminiLLM(api_key=args.gemini_api_key, model_name=DEFAULT_GEMINI_MODEL)
        else:
            llm = OllamaLLM(model_name=args.llm)
        
        selector = ClipSelector(llm)
        clips = selector.analyze_transcript(transcript, args.min_duration, args.max_duration)
        
        if not clips:
            print("No clips found. Try different parameters.")
            sys.exit(0)
        
        with open(clips_json_path, 'w', encoding='utf-8') as f:
            json.dump(clips, f, ensure_ascii=False, indent=2)
        print(f"\nClip selections saved to: {clips_json_path}")
    
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
    
    # Step 3: Extract clips
    if not args.extract_only:
        response = input("\nProceed with clip extraction? (y/n): ")
        if response.lower() != 'y':
            print("Clip extraction cancelled")
            return
    
    VideoClipper.create_clips(video_path, clips, str(output_dir))
    print(f"\n✓ All clips saved to: {output_dir}")
    
    # Final cleanup if requested
    if args.cleanup_audio and audio_path.exists():
        AudioConverter.cleanup_temp_audio(str(audio_path))


if __name__ == "__main__":
    main()
