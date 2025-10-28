"""
Gemini LLM provider
"""

import json
import sys
import subprocess
import requests
from typing import Dict, Optional

class GeminiLLM:
    """Interface with Google Gemini API for clip selection"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini client
        Args:
            api_key: Google AI API key
            model_name: Name of the Gemini model to use
        """
        try:
            from google import genai
        except ImportError:
            print("\n❌ google-genai package not installed")
            print("Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
            from google import genai
        
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        print(f"✓ Gemini API initialized ({model_name})")
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt]
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return ""
