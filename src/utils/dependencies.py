"""
Dependency checking utilities
"""

import sys
import subprocess
import shutil
import requests
from typing import Dict, List

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
            sys.exit(1)
        else:
            print("✓ ffmpeg is installed")
        
        # Check ffprobe
        if not DependencyChecker.check_ffprobe():
            print("❌ ffprobe is NOT installed")
            sys.exit(1)
        else:
            print("✓ ffprobe is installed")
        
        # Check Python packages
        required_packages = {
            "requests": "requests",
            "tqdm": "tqdm",
            "dotenv": "python-dotenv",
        }
        
        for module_name, package_name in required_packages.items():
            if not DependencyChecker.check_python_package(module_name):
                print(f"Installing {package_name}...")
                DependencyChecker.install_python_package(package_name)
            else:
                print(f"✓ {package_name} is installed")
        
        print("\n=== All dependencies satisfied ===\n")
    
    @staticmethod
    def check_and_install_specific(ctx: Dict) -> bool:
        """
        Check dependencies based on context.
        ctx contains: backend, llm_provider, operations, etc.
        """
        print("=== Checking Required Dependencies ===\n")
        
        all_passed = True
        
        # Check ffmpeg/ffprobe if needed
        if ctx.get('needs_ffmpeg', False):
            if not DependencyChecker.check_ffmpeg():
                print("❌ ffmpeg is NOT installed (required for video processing)")
                print("Please install ffmpeg:")
                print("  - Arch Linux: sudo pacman -S ffmpeg")
                print("  - Ubuntu/Debian: sudo apt install ffmpeg")
                print("  - macOS: brew install ffmpeg")
                print("  - Windows: choco install ffmpeg")
                all_passed = False
            else:
                print("✓ ffmpeg is installed")
            
            if not DependencyChecker.check_ffprobe():
                print("❌ ffprobe is NOT installed (required for video processing)")
                all_passed = False
            else:
                print("✓ ffprobe is installed")
        
        # Check Python packages
        required_modules = ctx.get('required_modules', [])
        for module_name in required_modules:
            if not DependencyChecker.check_python_package(module_name):
                # Try to install common packages
                package_map = {
                    'requests': 'requests',
                    'tqdm': 'tqdm',
                    'dotenv': 'python-dotenv',
                }
                if module_name in package_map:
                    print(f"⚠️  {module_name} not found, installing...")
                    try:
                        DependencyChecker.install_python_package(package_map[module_name])
                    except:
                        print(f"❌ Failed to install {module_name}")
                        all_passed = False
                else:
                    print(f"❌ {module_name} is required but not available")
                    all_passed = False
            else:
                print(f"✓ {module_name} is available")
        
        # Check Whisper if needed
        if ctx.get('needs_whisper', False):
            if not DependencyChecker.check_python_package('whisper'):
                print("❌ openai-whisper is NOT installed (required for Whisper backend)")
                print("Install with: pip install openai-whisper")
                all_passed = False
            else:
                print("✓ openai-whisper is available")
                
            # Check PyTorch
            if not DependencyChecker.check_python_package('torch'):
                print("❌ PyTorch is NOT installed (required for Whisper)")
                print("Install with: pip install torch")
                all_passed = False
            else:
                print("✓ PyTorch is available")
        
        # Check Gemini if needed
        if ctx.get('needs_gemini', False):
            if not DependencyChecker.check_python_package('google.genai'):
                print("❌ google-genai is NOT installed (required for Gemini)")
                print("Install with: pip install google-genai")
                all_passed = False
            else:
                print("✓ google-genai is available")
        
        # Check Ollama connection if needed
        if ctx.get('needs_ollama', False):
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✓ Ollama is running")
                else:
                    print("❌ Ollama is not responding correctly")
                    all_passed = False
            except requests.exceptions.ConnectionError:
                print("❌ Cannot connect to Ollama (required for --llm-provider ollama)")
                print("Please ensure Ollama is running: ollama serve")
                all_passed = False
        
        if all_passed:
            print("\n✓ All required dependencies satisfied\n")
        
        return all_passed
