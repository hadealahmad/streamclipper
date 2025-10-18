#!/usr/bin/env python3
"""
Video Clipper - Automatically extract clips from videos using AI
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
    # python-dotenv not installed, will try to install later
    pass

# Import configuration and prompts
from config import *
from prompts import GEMINI_TRANSCRIPTION_PROMPT, get_clip_selection_prompt


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
            print("  - Or download from: https://ffmpeg.org/download.html")
            sys.exit(1)
        else:
            print("✓ ffmpeg is installed")
        
        # Check ffprobe
        if not DependencyChecker.check_ffprobe():
            print("❌ ffprobe is NOT installed (usually comes with ffmpeg)")
            sys.exit(1)
        else:
            print("✓ ffprobe is installed")
        
        # Check and install Python packages
        required_packages = {
            "requests": "requests",
            "tqdm": "tqdm",
            "dotenv": "python-dotenv",
        }
        
        # faster-whisper is optional (only needed for local transcription)
        optional_packages = {
            "faster_whisper": "faster-whisper",
        }
        
        for module_name, package_name in required_packages.items():
            if not DependencyChecker.check_python_package(module_name):
                print(f"Installing {package_name}...")
                DependencyChecker.install_python_package(package_name)
            else:
                print(f"✓ {package_name} is installed")
        
        # Check optional packages
        for module_name, package_name in optional_packages.items():
            if DependencyChecker.check_python_package(module_name):
                print(f"✓ {package_name} is installed")
            else:
                print(f"ℹ {package_name} not installed (optional, for local transcription)")
        
        print("\n=== All dependencies satisfied ===\n")


class GeminiTranscriber:
    """Transcribe video using Gemini API"""
    
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
        # Use Gemini 2.0 Flash which supports video/audio
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
        
        # Upload the video file
        video_file = genai.upload_file(path=video_path)
        
        print("Processing transcription with Gemini...")
        
        prompt = GEMINI_TRANSCRIPTION_PROMPT

        response = self.model.generate_content([video_file, prompt])
        
        # Parse the response
        transcript_segments = self._parse_gemini_response(response.text)
        
        print(f"✓ Transcription complete: {len(transcript_segments)} segments")
        
        return transcript_segments
    
    def _parse_gemini_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini's CSV transcription response"""
        import csv
        import io
        
        # Try to parse as CSV first
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
        
        # Fallback: try to parse as JSON (for backward compatibility)
        import re
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
        if json_match:
            try:
                segments = json.loads(json_match.group())
                
                # Validate segments
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
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON response: {e}")
        
        # Final fallback: try to parse line by line
        print("Warning: Using fallback parser for transcription")
        segments = []
        current_time = 0.0
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('[') and not line.startswith('start_time'):
                # Estimate segment duration (about 5 seconds per line)
                segments.append({
                    'start': current_time,
                    'end': current_time + 5.0,
                    'text': line
                })
                current_time += 5.0
        
        return segments if segments else []


class VideoTranscriber:
    """Transcribe video using Whisper (faster-whisper backend)"""
    
    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL, use_gpu: bool = DEFAULT_USE_GPU):
        """
        Initialize transcriber
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            use_gpu: Whether to use GPU for transcription
        """
        from faster_whisper import WhisperModel
        
        print(f"Loading Whisper model ({model_size})...")
        
        # Determine device and compute type
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    # Try different compute types for different GPU types
                    compute_type = "float16"
                    print("GPU detected, attempting to use float16")
                else:
                    device = "cpu"
                    compute_type = "int8"
                    print("GPU not available (PyTorch CPU-only build), using CPU")
                    print("For AMD GPU support, install PyTorch with ROCm:")
                    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6")
            except ImportError:
                device = "cpu"
                compute_type = "int8"
                print("PyTorch not available, using CPU")
        else:
            device = "cpu"
            compute_type = "int8"
            print("Using CPU for transcription")
        
        # Try to create model with different configurations
        model_created = False
        
        if device == "cuda":
            # Try float16 first (NVIDIA GPUs)
            try:
                self.model = WhisperModel(model_size, device=device, compute_type="float16")
                print("Model loaded successfully with float16 (NVIDIA GPU)")
                model_created = True
            except ValueError as e:
                if "float16" in str(e):
                    print("Float16 not supported, trying int8...")
                else:
                    raise e
            
            # Try int8 if float16 failed (AMD GPUs or older NVIDIA)
            if not model_created:
                try:
                    self.model = WhisperModel(model_size, device=device, compute_type="int8")
                    print("Model loaded successfully with int8 (AMD GPU or older NVIDIA)")
                    model_created = True
                except Exception as e:
                    print(f"GPU failed, falling back to CPU: {e}")
                    device = "cpu"
                    compute_type = "int8"
        
        # Fallback to CPU if GPU failed
        if not model_created:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("Model loaded successfully on CPU")
    
    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Transcribe video with timestamps
        Returns list of segments with text, start, and end times
        """
        print(f"Transcribing: {video_path}")
        
        segments, info = self.model.transcribe(
            video_path,
            language="ar",  # Arabic
            vad_filter=True,  # Voice activity detection
            word_timestamps=True
        )
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        transcript_segments = []
        from tqdm import tqdm
        
        for segment in tqdm(segments, desc="Processing segments"):
            transcript_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        return transcript_segments


class VideoTranscriberROCm:
    """Transcribe video using original Whisper (supports AMD ROCm)"""
    
    def __init__(self, model_size: str = "medium", gpu_device: Optional[int] = None):
        """
        Initialize transcriber with PyTorch backend (AMD GPU compatible)
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            gpu_device: GPU device index (None = auto-detect, -1 = force CPU)
        """
        import whisper
        import torch
        
        print(f"Loading Whisper model ({model_size})...")
        
        # Detect device
        if gpu_device == -1:
            device = "cpu"
            print("Using CPU (forced)")
        elif torch.cuda.is_available():
            if gpu_device is not None:
                if gpu_device >= torch.cuda.device_count():
                    print(f"Warning: GPU {gpu_device} not available. Available: {torch.cuda.device_count()}")
                    device = "cpu"
                    print("Falling back to CPU")
                else:
                    device = f"cuda:{gpu_device}"
                    print(f"Using GPU {gpu_device}: {torch.cuda.get_device_name(gpu_device)}")
            else:
                # Auto-detect: list all GPUs and use device 0
                print(f"Available GPUs ({torch.cuda.device_count()}):")
                for i in range(torch.cuda.device_count()):
                    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                device = "cuda:0"
                print(f"Auto-selected GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("No GPU available, using CPU")
        
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        print("Model loaded successfully")
    
    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Transcribe video with timestamps
        Returns list of segments with text, start, and end times
        """
        print(f"Transcribing: {video_path}")
        
        result = self.model.transcribe(
            video_path,
            language="ar",  # Arabic
            word_timestamps=False,  # Use sentence-level timestamps
            verbose=False
        )
        
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


class GeminiLLM:
    """Interface with Google Gemini API"""
    
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
    """Interface with local Ollama LLM"""
    
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
        
        # Check if Ollama is running
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
                    print(f"\nTo install the model, run:")
                    print(f"  ollama pull {self.model_name}")
                    
                    response = input("\nDo you want to continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        sys.exit(1)
            else:
                print("✓ Ollama is running")
        except requests.exceptions.ConnectionError:
            print("\n❌ Cannot connect to Ollama")
            print("Please ensure Ollama is installed and running:")
            print("  - Install: https://ollama.ai/download")
            print("  - Start: ollama serve")
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
    
    def analyze_transcript(self, transcript: List[Dict], min_duration: float = 10, max_duration: float = 60) -> List[Dict]:
        """
        Analyze transcript and select clips
        Returns list of clips with start, end, reason
        """
        print("\nAnalyzing transcript for interesting clips...")
        
        # Format transcript for LLM
        transcript_text = self._format_transcript(transcript)
        
        prompt = get_clip_selection_prompt(transcript_text, min_duration, max_duration)

        response = self.llm.generate(prompt)
        
        # Extract JSON from response
        clips = self._extract_clips_from_response(response, transcript, min_duration, max_duration)
        
        print(f"Found {len(clips)} potential clips")
        return clips
    
    def _format_transcript(self, transcript: List[Dict]) -> str:
        """Format transcript with timestamps"""
        lines = []
        for seg in transcript:
            lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
        return "\n".join(lines)
    
    def _extract_clips_from_response(self, response: str, transcript: List[Dict], min_duration: float, max_duration: float) -> List[Dict]:
        """Extract and validate clips from LLM response"""
        # Try to find JSON array in response
        import re
        
        # Look for JSON array
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            try:
                clips = json.loads(json_match.group())
                
                # Validate and filter clips
                valid_clips = []
                video_duration = transcript[-1]['end'] if transcript else 0
                
                for clip in clips:
                    if not all(key in clip for key in ['start', 'end', 'reason']):
                        continue
                    
                    start = float(clip['start'])
                    end = float(clip['end'])
                    duration = end - start
                    
                    # Validate clip
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
            "-y",  # Overwrite output
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


def save_transcript(transcript: List[Dict], output_path: str, format_type: str = DEFAULT_TRANSCRIPT_FORMAT):
    """Save transcript to file (CSV by default)"""
    if format_type == "csv":
        csv_path = output_path.replace('.json', '.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start_time', 'end_time', 'text'])
            for segment in transcript:
                writer.writerow([segment['start'], segment['end'], segment['text']])
        print(f"Transcript saved to: {csv_path}")
        
        # Also save as JSON for backward compatibility
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"Transcript also saved as JSON: {output_path}")
    else:
        # Save as JSON only
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"Transcript saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automatically extract clips from videos using AI"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for clips")
    parser.add_argument("-m", "--model", default=DEFAULT_WHISPER_MODEL, 
                       choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                       help="Whisper model size")
    parser.add_argument("--backend", default=DEFAULT_TRANSCRIPTION_BACKEND,
                       choices=["faster-whisper", "openai-whisper", "gemini"],
                       help="Transcription backend (gemini for cloud-based, no model download)")
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU device index to use (0, 1, etc). Use -1 for CPU. Omit for auto-detect.")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU usage (disable GPU)")
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, 
                       choices=["gemini", "ollama"],
                       help="LLM provider for clip selection")
    parser.add_argument("--llm", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name (if using ollama)")
    parser.add_argument("--gemini-api-key", default=None,
                       help="Google Gemini API key (default: from .env file)")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION, help="Minimum clip duration (seconds)")
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION, help="Maximum clip duration (seconds)")
    parser.add_argument("--long-form", action="store_true", help="Optimize for long-form videos (10+ min clips). Sets min=600s, max=1800s")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription (use existing transcript.json)")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip LLM analysis (use existing clips.json)")
    parser.add_argument("--transcript-format", default=DEFAULT_TRANSCRIPT_FORMAT, choices=["csv", "json"],
                       help="Transcript output format")
    
    args = parser.parse_args()
    
    # Apply long-form preset if specified
    if args.long_form:
        args.min_duration = 600  # 10 minutes
        args.max_duration = 1800  # 30 minutes
        print("Long-form mode: Looking for 10-30 minute segments")
    
    # Load API key from environment if not provided
    if not args.gemini_api_key:
        args.gemini_api_key = os.getenv(GEMINI_API_KEY_ENV)
        if not args.gemini_api_key:
            print("\n⚠️  Warning: No Gemini API key found")
            print("Please either:")
            print("  1. Set GEMINI_API_KEY in .env file")
            print("  2. Use --gemini-api-key argument")
            print("  3. Use --llm-provider ollama for local processing")
            if args.llm_provider == "gemini" or args.backend == "gemini":
                print("\nError: Gemini API key required for selected options")
                sys.exit(1)
    
    # Handle GPU setting
    use_gpu = DEFAULT_USE_GPU and not args.no_gpu
    if args.gpu is not None:
        use_gpu = args.gpu != -1
    
    # Check dependencies
    DependencyChecker.check_and_install_dependencies()
    
    # Validate video file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    video_path = args.video
    video_name = Path(video_path).stem
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    transcript_path = output_dir / f"{video_name}_transcript.json"
    
    # Step 1: Transcribe video
    if args.skip_transcribe:
        # Try to load existing transcript
        if transcript_path.exists():
            print(f"Loading existing transcript: {transcript_path}")
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        else:
            # Try to load CSV version
            csv_path = str(transcript_path).replace('.json', '.csv')
            if os.path.exists(csv_path):
                print(f"Loading existing transcript from CSV: {csv_path}")
                transcript = []
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        transcript.append({
                            'start': float(row['start_time']),
                            'end': float(row['end_time']),
                            'text': row['text']
                        })
            else:
                print("No existing transcript found, will transcribe...")
                transcript = None
    else:
        transcript = None
    
    if transcript is None:
        # Select transcriber backend
        if args.backend == "gemini":
            transcriber = GeminiTranscriber(api_key=args.gemini_api_key)
        elif args.backend == "openai-whisper":
            transcriber = VideoTranscriberROCm(model_size=args.model, gpu_device=args.gpu)
        else:
            transcriber = VideoTranscriber(model_size=args.model, use_gpu=use_gpu)
        
        transcript = transcriber.transcribe(video_path)
        save_transcript(transcript, str(transcript_path), args.transcript_format)
    
    if not transcript:
        print("Error: No transcript generated")
        sys.exit(1)
    
    # Step 2: Analyze transcript and select clips
    clips_json_path = output_dir / f"{video_name}_clips.json"
    
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
            print("No clips found. Try a different video or adjust parameters.")
            sys.exit(0)
        
        # Save clip selections
        with open(clips_json_path, 'w', encoding='utf-8') as f:
            json.dump(clips, f, ensure_ascii=False, indent=2)
        print(f"\nClip selections saved to: {clips_json_path}")
    
    # Display clips
    print("\n=== Selected Clips ===")
    for i, clip in enumerate(clips, 1):
        duration = clip['end'] - clip['start']
        print(f"\nClip {i}:")
        print(f"  Time: {clip['start']:.1f}s - {clip['end']:.1f}s ({duration:.1f}s)")
        print(f"  Reason: {clip['reason']}")
    
    # Step 3: Extract clips
    response = input("\nProceed with clip extraction? (y/n): ")
    if response.lower() == 'y':
        VideoClipper.create_clips(video_path, clips, str(output_dir))
        print(f"\n✓ All clips saved to: {output_dir}")
    else:
        print("Clip extraction cancelled")


if __name__ == "__main__":
    main()

