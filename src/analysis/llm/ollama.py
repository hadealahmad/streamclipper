"""
Ollama LLM provider
"""

import json
import sys
import subprocess
import requests
from typing import Dict, Optional

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
