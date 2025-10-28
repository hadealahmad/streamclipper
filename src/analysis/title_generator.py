"""
Clip title generation using LLM
"""

import json
import re
from typing import List, Dict

from prompts import (
    get_clip_selection_prompt,
    get_timestamp_generation_prompt,
    get_clip_title_generation_prompt
)

class ClipTitleGenerator:
    """Generate suggested titles for video clips"""
    
    def __init__(self, llm):
        """
        Initialize clip title generator
        Args:
            llm: LLM instance (GeminiLLM, OllamaLLM, or OpenRouterLLM)
        """
        self.llm = llm
    
    def generate_titles(self, clip: Dict, transcript: List[Dict]) -> Dict[str, str]:
        """
        Generate suggested titles for a clip
        Returns dict with 'arabic' and 'english' titles
        """
        # Extract clip content from transcript
        clip_content = self._extract_clip_content(clip, transcript)
        clip_duration = clip['end'] - clip['start']
        
        prompt = get_clip_title_generation_prompt(clip_content, clip['reason'], clip_duration)
        response = self.llm.generate(prompt)
        titles = self._extract_titles_from_response(response)
        
        return titles
    
    def _extract_clip_content(self, clip: Dict, transcript: List[Dict]) -> str:
        """Extract text content for a specific clip from transcript"""
        content_segments = []
        for segment in transcript:
            # Check if segment overlaps with clip
            if (segment['start'] < clip['end'] and segment['end'] > clip['start']):
                content_segments.append(segment['text'])
        
        return ' '.join(content_segments)
    
    def _extract_titles_from_response(self, response: str) -> Dict[str, str]:
        """Extract titles from LLM response"""
        import re
        
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                titles = json.loads(json_match.group())
                if 'arabic' in titles and 'english' in titles:
                    return {
                        'arabic': str(titles['arabic']).strip(),
                        'english': str(titles['english']).strip()
                    }
            except json.JSONDecodeError:
                pass
        
        print("Warning: Could not parse title response properly")
        return {
            'arabic': 'عنوان مقترح',
            'english': 'Suggested Title'
        }
