"""
Base transcription interface
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseTranscriber(ABC):
    """Abstract base class for transcription backends"""
    
    @abstractmethod
    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Transcribe video with timestamps
        Returns list of segments with text, start, and end times
        """
        pass
