"""
Tests for the main Pi Detector application.
"""

import pytest
from unittest.mock import Mock, patch
from pi_detector.main import PiDetectorApp


class TestPiDetectorApp:
    """Test cases for PiDetectorApp."""
    
    def test_app_initialization(self):
        """Test app can be initialized."""
        app = PiDetectorApp()
        assert app is not None
        assert app.config is not None
    
    def test_format_detection_message(self):
        """Test detection message formatting."""
        app = PiDetectorApp()
        
        # Test person detection
        msg = app._format_detection_message("person", 0.85)
        assert "Human" in msg
        
        # Test animal detection
        msg = app._format_detection_message("dog", 0.75)
        assert "Dog" in msg
    
    @patch('pi_detector.main.CameraHandler')
    @patch('pi_detector.main.ObjectDetector')
    @patch('pi_detector.main.AudioOutputSystem')
    def test_initialization_success(self, mock_audio, mock_detector, mock_camera):
        """Test successful initialization of all components."""
        app = PiDetectorApp()
        result = app.initialize()
        
        assert result is True
        assert mock_camera.called
        assert mock_detector.called
        assert mock_audio.called
