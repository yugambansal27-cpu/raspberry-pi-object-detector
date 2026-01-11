import os
import subprocess
import sys
from pathlib import Path

def create_virtual_environment(venv_name):
    """Create a new virtual environment with the specified name"""
    try:
        # Create the virtual environment
        subprocess.run([sys.executable, '-m', 'venv', venv_name], check=True)
        
        # Get the activation script path based on OS
        if os.name == 'nt':  # Windows
            activate_script = Path(venv_name) / 'Scripts' / 'activate.bat'
        else:  # Unix-like
            activate_script = Path(venv_name) / 'bin' / 'activate'
            
        print(f"Virtual environment '{venv_name}' created successfully!")
        print(f"\nTo activate the virtual environment:")
        if os.name == 'nt':
            print(f"    {activate_script}")
        else:
            print(f"    source {activate_script}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    venv_name = "myenv"  # Default name
    if len(sys.argv) > 1:
        venv_name = sys.argv[1]
    create_virtual_environment(venv_name)
