"""
Timestamp generation using LLM
"""

import json
import re
from typing import List, Dict

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from prompts import (
    get_timestamp_generation_prompt,
)

class TimestampGenerator:
    """Generate video section timestamps with Arabic titles"""
    
    def __init__(self, llm):
        """
        Initialize timestamp generator
        Args:
            llm: LLM instance (GeminiLLM, OllamaLLM, or OpenRouterLLM)
        """
        self.llm = llm
    
    def generate_timestamps(self, transcript: List[Dict], video_duration: float) -> List[Dict]:
        """
        Generate video section timestamps
        Returns list of timestamp sections with Arabic titles
        """
        print("\nGenerating video section timestamps...")
        
        transcript_text = self._format_transcript(transcript)
        prompt = get_timestamp_generation_prompt(transcript_text, video_duration)
        response = self.llm.generate(prompt)
        timestamps = self._extract_timestamps_from_response(response, video_duration)
        
        print(f"Generated {len(timestamps)} timestamp sections")
        return timestamps
    
    def _format_transcript(self, transcript: List[Dict]) -> str:
        """Format transcript with timestamps for timestamp generation"""
        lines = []
        for seg in transcript:
            lines.append(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
        return "\n".join(lines)
    
    def _extract_timestamps_from_response(self, response: str, video_duration: float) -> List[Dict]:
        """Extract and validate timestamps from LLM response"""
        import re
        
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            try:
                timestamps = json.loads(json_match.group())
                valid_timestamps = []
                
                for timestamp in timestamps:
                    if not all(key in timestamp for key in ['section', 'start_time', 'end_time', 'arabic_title']):
                        continue
                    
                    start_time = float(timestamp['start_time'])
                    end_time = float(timestamp['end_time'])
                    section = int(timestamp['section'])
                    arabic_title = str(timestamp['arabic_title']).strip()
                    
                    if (start_time >= 0 and end_time <= video_duration and 
                        start_time < end_time and arabic_title):
                        valid_timestamps.append({
                            'section': section,
                            'start_time': start_time,
                            'end_time': end_time,
                            'arabic_title': arabic_title
                        })
                
                # Sort by section number and validate continuity
                valid_timestamps.sort(key=lambda x: x['section'])
                
                # Validate that sections are continuous and cover the full video
                if valid_timestamps:
                    if valid_timestamps[0]['start_time'] != 0.0:
                        print("Warning: First section doesn't start at 0.0")
                    
                    if abs(valid_timestamps[-1]['end_time'] - video_duration) > 1.0:
                        print(f"Warning: Last section doesn't end at video duration ({video_duration:.1f})")
                    
                    # Check for gaps or overlaps
                    for i in range(len(valid_timestamps) - 1):
                        current_end = valid_timestamps[i]['end_time']
                        next_start = valid_timestamps[i + 1]['start_time']
                        if abs(current_end - next_start) > 0.1:  # Allow small gaps
                            print(f"Warning: Gap between sections {i+1} and {i+2}")
                
                return valid_timestamps
            except json.JSONDecodeError:
                pass
        
        print("Warning: Could not parse timestamp response properly")
        return []
