import subprocess
import sys

# List of required libraries
required_libraries = [
    "opencv-python",          # For cv2
    "mss",                    # For screen capturing
    "pyserial",               # For serial communication
    "numpy",                  # For numerical operations
    "matplotlib",             # For plotting
    "scipy",                  # For signal processing
    "sounddevice",            # For audio capture
]

# Function to install packages
def install_packages(packages):
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

# Import hamming function after ensuring scipy is installed
def import_hamming():
    try:
        from scipy.signal.windows import hamming
        print("Successfully imported hamming from scipy.signal.windows.")
    except ImportError as e:
        print(f"Error importing hamming: {e}. Ensure scipy is updated.")

# Run the installation and import
if __name__ == "__main__":
    install_packages(required_libraries)
    import_hamming()
