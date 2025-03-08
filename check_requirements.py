"""
Check Requirements for nGPT Explorer

This script checks if all required packages are installed and provides helpful error messages.
"""

import importlib.util
import sys
import subprocess
import os

# Required packages
REQUIRED_PACKAGES = [
    "torch",
    "pygame",
    "matplotlib",
    "numpy",
    "einops",
    "sklearn",
    "rotary_embedding_torch"
]

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Attempt to install a package"""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Check all required packages"""
    print("Checking required packages for nGPT Explorer...")
    print()
    
    all_installed = True
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        sys.stdout.write(f"Checking {package}... ")
        sys.stdout.flush()
        
        if check_package(package):
            print("OK")
        else:
            print("MISSING")
            missing_packages.append(package)
            all_installed = False
    
    print()
    
    if all_installed:
        print("All required packages are installed!")
        print("You're ready to run the nGPT Explorer.")
    else:
        print("The following packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        
        print()
        print("Would you like to install the missing packages? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            print("Installing missing packages...")
            
            for package in missing_packages:
                if install_package(package):
                    print(f"Successfully installed {package}")
                else:
                    print(f"Failed to install {package}")
            
            # Check again
            still_missing = []
            for package in missing_packages:
                if not check_package(package):
                    still_missing.append(package)
            
            if still_missing:
                print()
                print("Some packages could not be installed automatically.")
                print("Please install them manually using:")
                print("pip install -r requirements.txt")
            else:
                print()
                print("All packages successfully installed!")
                print("You're ready to run the nGPT Explorer.")
        else:
            print()
            print("Please install the missing packages using:")
            print("pip install -r requirements.txt")
    
    # Special check for CUDA if torch is installed
    if check_package("torch"):
        import torch
        if torch.cuda.is_available():
            print()
            print("CUDA is available! The explorer will use GPU acceleration.")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print()
            print("CUDA is not available. The explorer will run on CPU only.")
            print("This is fine for exploration, but may be slower for larger models.")

if __name__ == "__main__":
    # Clear screen for better visibility
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== nGPT Explorer Requirements Check ===")
    print()
    main() 