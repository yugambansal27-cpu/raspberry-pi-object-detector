"""
Raspberry Pi AI Camera Detection System
A Python application for detecting humans and animals with audio output.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .main import main
from .detector import ObjectDetector
from .camera import CameraHandler
from .audio import AudioOutputSystem

__all__ = ["main", "ObjectDetector", "CameraHandler", "AudioOutputSystem"]
