#!/usr/bin/env python3
"""
Test script to verify all components are working
Run this after installation to ensure everything is set up correctly
"""

import sys
import subprocess
import shutil


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def test_ffmpeg():
    """Test if ffmpeg is installed and working"""
    print_header("Testing ffmpeg")
    
    if not shutil.which("ffmpeg"):
        print("✗ ffmpeg NOT found")
        print("  Install: sudo pacman -S ffmpeg")
        return False
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"✓ ffmpeg found: {version_line}")
        return True
    except Exception as e:
        print(f"✗ ffmpeg error: {e}")
        return False


def test_ffprobe():
    """Test if ffprobe is installed"""
    print_header("Testing ffprobe")
    
    if not shutil.which("ffprobe"):
        print("✗ ffprobe NOT found")
        return False
    
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"✓ ffprobe found: {version_line}")
        return True
    except Exception as e:
        print(f"✗ ffprobe error: {e}")
        return False


def test_python_packages():
    """Test if Python packages are installed"""
    print_header("Testing Python Packages")
    
    packages = {
        "faster_whisper": "faster-whisper",
        "requests": "requests",
        "tqdm": "tqdm",
    }
    
    all_installed = True
    
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} NOT installed")
            print(f"  Install: pip install {package_name}")
            all_installed = False
    
    return all_installed


def test_gemini():
    """Test if Gemini API package is installed"""
    print_header("Testing Gemini API")
    
    try:
        import google.generativeai as genai
        print("✓ google-generativeai package is installed")
        print("  Using Gemini for clip selection (default)")
        return True
    except ImportError:
        print("⚠ google-generativeai not installed")
        print("  Will be auto-installed on first run")
        print("  Or install: pip install google-generativeai")
        return True  # Not critical, will auto-install


def test_ollama():
    """Test if Ollama is running and accessible (optional)"""
    print_header("Testing Ollama (Optional)")
    
    print("Note: Ollama is optional. Script uses Gemini by default.")
    
    try:
        import requests
        
        # Test connection
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        
        if response.status_code != 200:
            print("⚠ Ollama responded with error")
            print("  Not required - Gemini is the default LLM")
            return True
        
        print("✓ Ollama is running (available as alternative)")
        
        # List models
        models = response.json().get("models", [])
        
        if not models:
            print("⚠ No models installed in Ollama")
            print("  Not required - Gemini is the default")
        else:
            print(f"✓ Found {len(models)} Ollama model(s)")
        
        return True
        
    except ImportError:
        print("✗ 'requests' package not installed")
        return False
    except Exception:
        print("⚠ Ollama not running")
        print("  Not required - Gemini is the default LLM")
        print("  To use Ollama: ollama serve, then --llm-provider ollama")
        return True  # Not critical


def test_whisper_model():
    """Test if Whisper can be loaded (doesn't download full model)"""
    print_header("Testing Whisper (Quick Check)")
    
    try:
        from faster_whisper import WhisperModel
        print("✓ faster-whisper package is working")
        print("  Note: Models will be downloaded on first use (~1-2GB)")
        return True
    except ImportError:
        print("✗ faster-whisper not installed")
        print("  Install: pip install faster-whisper")
        return False
    except Exception as e:
        print(f"✗ Error loading faster-whisper: {e}")
        return False


def test_video_clipper_script():
    """Test if the main script is present and can show help"""
    print_header("Testing video_clipper.py Script")
    
    try:
        result = subprocess.run(
            [sys.executable, "video_clipper.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✓ video_clipper.py script is working")
            return True
        else:
            print("✗ video_clipper.py script failed")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("✗ video_clipper.py not found in current directory")
        return False
    except Exception as e:
        print(f"✗ Error testing script: {e}")
        return False


def print_summary(results):
    """Print test summary"""
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTests passed: {passed}/{total}\n")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("  ✓ ALL TESTS PASSED - System is ready!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Read README.md for usage guide")
        print("  2. Try: python video_clipper.py --help")
        print("  3. Process a video: python video_clipper.py your_video.mp4")
        return True
    else:
        print("\n" + "=" * 60)
        print("  ✗ SOME TESTS FAILED - Fix issues above")
        print("=" * 60)
        print("\nRefer to README.md for detailed setup guide")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  Video Clipper - Setup Test")
    print("=" * 60)
    print("\nThis script will verify your installation is correct.\n")
    
    results = {}
    
    # Run tests
    results["ffmpeg"] = test_ffmpeg()
    results["ffprobe"] = test_ffprobe()
    results["Python packages"] = test_python_packages()
    results["Gemini API"] = test_gemini()
    results["Ollama (optional)"] = test_ollama()
    results["Whisper"] = test_whisper_model()
    results["Main script"] = test_video_clipper_script()
    
    # Print summary
    success = print_summary(results)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

