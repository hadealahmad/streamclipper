"""
Clip selection using LLM analysis
"""

import json
import re
from typing import List, Dict

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from prompts import (
    get_clip_selection_prompt,
    get_timestamp_generation_prompt,
    get_clip_title_generation_prompt
)

class ClipSelector:
    """Use LLM to select interesting clips from transcript"""
    
    def __init__(self, llm):
        """
        Initialize clip selector
        Args:
            llm: LLM instance (GeminiLLM, OllamaLLM, or OpenRouterLLM)
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
