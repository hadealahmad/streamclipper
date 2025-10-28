"""
Base LLM interface
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(selfblogs, prompt: str) -> str:
        """Generate response from LLM"""
        pass
