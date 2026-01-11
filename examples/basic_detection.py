"""
Basic example of using the Pi Detector system.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pi_detector.main import PiDetectorApp


def main():
    """Run basic detection example."""
    print("=" * 60)
    print("Raspberry Pi AI Camera Detection System - Basic Example")
    print("=" * 60)
    print("\nThis example demonstrates real-time detection of humans and animals.")
    print("\nPress Ctrl+C to stop.\n")
    
    # Create and run app
    app = PiDetectorApp(config_path="config/settings.json")
    app.run()


if __name__ == "__main__":
    main()
