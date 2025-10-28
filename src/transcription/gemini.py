"""
Gemini transcription backend
"""

import json
import sys
import subprocess
import os
from typing import List, Dict, Optional

from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from prompts import GEMINI_TRANSCRIPTION_PROMPT
from src.extraction.audio_converter import AudioConverter

class GeminiTranscriber:
    """Transcribe video using Gemini API (cloud-based)"""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini transcriber
        Args:
            api_key: Google AI API key
        """
        try:
            from google import genai
        except ImportError:
            print("\n❌ google-genai package not installed")
            print("Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
            from google import genai
        
        self.client = genai.Client(api_key=api_key)
        print("✓ Gemini transcriber initialized")
    
    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Transcribe video with timestamps using Gemini
        Returns list of segments with text, start, and end times
        """
        print(f"Converting video to MP3 for faster upload...")
        
        # Convert video to MP3 first to reduce upload size
        audio_path = AudioConverter.video_to_mp3(video_path)
        
        print(f"Uploading audio to Gemini: {audio_path}")
        
        # Check file size
        import os
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"Audio file size: {file_size_mb:.2f} MB")
        print("This may take a few minutes depending on audio size...")
        
        # Upload the audio file using the new API
        try:
            audio_file = self.client.files.upload(file=audio_path)
            print(f"✓ Upload successful. File URI: {audio_file.uri if hasattr(audio_file, 'uri') else 'N/A'}")
            print(f"  File name: {audio_file.name if hasattr(audio_file, 'name') else 'N/A'}")
            print(f"  File state: {audio_file.state if hasattr(audio_file, 'state') else 'N/A'}")
        except Exception as e:
            print(f"Error uploading file to Gemini: {e}")
            return []
        
        print("Processing transcription with Gemini...")
        
        # Generate content using the new API
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[GEMINI_TRANSCRIPTION_PROMPT, audio_file]
            )
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return []
        
        # Debug: Print response details
        print(f"Response received. Type: {type(response)}")
        if hasattr(response, 'text'):
            print(f"Response text length: {len(response.text) if response.text else 0}")
            if response.text:
                print(f"Response preview: {response.text[:200]}...")
        
        # Check for safety blocks
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'finish_reason'):
                    print(f"Finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings'):
                    print(f"Safety ratings: {candidate.safety_ratings}")
        
        # Check if response is valid
        if not response or not response.text:
            print("Error: Gemini returned empty or blocked response")
            print("This could be due to:")
            print("  - Content safety filters")
            print("  - API quota exceeded")
            print("  - Network issues")
            print("  - Audio format not supported")
            return []
        
        transcript_segments = self._parse_gemini_response(response.text)
        
        print(f"✓ Transcription complete: {len(transcript_segments)} segments")
        
        # Keep the MP3 file for potential reuse (don't clean up automatically)
        print(f"Audio file kept for reuse: {audio_path}")
        
        return transcript_segments
    
    def _parse_gemini_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini's CSV transcription response"""
        import csv
        import io
        
        # Validate input
        if not response_text or not isinstance(response_text, str):
            print(f"Warning: Invalid response_text type: {type(response_text)}")
            return []
        
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
