"""
Dependency checking utilities
"""

import sys
import subprocess
import shutil

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
