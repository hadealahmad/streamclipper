"""
OpenRouter LLM provider
"""

import json
import sys
import subprocess
import requests
from typing import Dict, Optional

class OpenRouterLLM:
    """Interface with Open Router API for LLM access"""
    
    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-3.3-70b-instruct:free"):
        """
        Initialize Open Router client
        Args:
            api_key: Open Router API key
            model_name: Name of the model to use (e.g., "meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-4")
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        print(f"✓ Open Router API initialized ({model_name})")
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the message content from the response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                print(f"Error: Unexpected response format from Open Router")
                return ""
        except requests.exceptions.RequestException as e:
            print(f"Error calling Open Router API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return ""
