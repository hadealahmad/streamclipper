import os
from typing import List, Dict, Optional, Tuple

from src.config import GPU_DEVICE_ENV

class GPUDetector:
    """Detect and manage GPU devices for dual/multi-GPU setups"""
    
    @staticmethod
    def detect_gpus() -> Tuple[List[Dict], str]:
        """
        Detect available GPUs and return device info
        Returns: (list of GPU info dicts, device type string)
        """
        try:
            import torch
            
            if not torch.cuda.is_available():
                return [], "cpu"
            
            gpu_count = torch.cuda.device_count()
            gpus = []
            
            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                
                gpus.append({
                    'id': i,
                    'name': device_name,
                    'memory': device_props.total_memory / (1024**3),  # GB
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
            
            # Detect if ROCm or CUDA
            device_type = "rocm" if "AMD" in gpus[0]['name'] or "Radeon" in gpus[0]['name'] else "cuda"
            
            return gpus, device_type
            
        except ImportError:
            return [], "cpu"
    
    @staticmethod
    def select_gpu(gpu_device: Optional[int] = None, interactive: bool = False) -> Tuple[str, int]:
        """
        Select GPU device
        Returns: (device string, device id)
        """
        gpus, device_type = GPUDetector.detect_gpus()
        
        if device_type == "cpu":
            print("No GPU detected or PyTorch not installed")
            print("\nFor AMD GPU support, install:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
            print("\nFor NVIDIA GPU support, install:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return "cpu", -1
        
        # Check environment variable
        if gpu_device is None:
            env_gpu = os.getenv(GPU_DEVICE_ENV)
            if env_gpu is not None:
                try:
                    gpu_device = int(env_gpu)
                    print(f"Using GPU {gpu_device} from environment variable {GPU_DEVICE_ENV}")
                except ValueError:
                    pass
        
        # Display available GPUs
        print(f"\n{'='*60}")
        print(f"GPU Detection - {device_type.upper()} Backend")
        print(f"{'='*60}")
        print(f"\nFound {len(gpus)} GPU(s):\n")
        
        for gpu in gpus:
            print(f"  [{gpu['id']}] {gpu['name']}")
            print(f"      Memory: {gpu['memory']:.1f} GB")
            print(f"      Compute: {gpu['compute_capability']}")
            print()
        
        # GPU selection logic
        if gpu_device is not None:
            if gpu_device == -1:
                print("CPU mode forced by user")
                return "cpu", -1
            elif 0 <= gpu_device < len(gpus):
                selected = gpus[gpu_device]
                print(f"Selected GPU {gpu_device}: {selected['name']}")
                return f"cuda:{gpu_device}", gpu_device
            else:
                print(f"Warning: GPU {gpu_device} not available")
                gpu_device = None
        
        # Interactive selection for dual+ GPU setups
        if interactive and len(gpus) > 1:
            print("Multiple GPUs detected.")
            while True:
                try:
                    choice = input(f"Select GPU [0-{len(gpus)-1}] or 'c' for CPU: ").strip().lower()
                    if choice == 'c':
                        return "cpu", -1
                    device_id = int(choice)
                    if 0 <= device_id < len(gpus):
                        selected = gpus[device_id]
                        print(f"Selected GPU {device_id}: {selected['name']}")
                        return f"cuda:{device_id}", device_id
                except (ValueError, KeyboardInterrupt):
                    pass
                print("Invalid selection. Try again.")
        
        # Auto-select first GPU
        selected = gpus[0]
        print(f"Auto-selected GPU 0: {selected['name']}")
        return "cuda:0", 0

