import subprocess
import sys
import os

def check_dependencies():
    required_packages = [
        'opencv-python',
        'numpy',
        'tensorflow',
        'batch-face',
        'keras'
    ]
    
    print("Checking dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is not installed")
    
    if missing_packages:
        print("\nInstalling missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All dependencies installed successfully!")
    else:
        print("\nAll dependencies are already installed!")

if __name__ == "__main__":
    check_dependencies() 