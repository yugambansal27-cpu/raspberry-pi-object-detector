"""
Main application entry point for the Pi Detector system.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

from .config import Config
from .camera import CameraHandler
from .detector import ObjectDetector
from .audio import AudioOutputSystem


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pi_detector.log')
    ]
)

logger = logging.getLogger(__name__)


class PiDetectorApp:
    """Main application class for Pi Detector."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Pi Detector application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.camera = None
        self.detector = None
        self.audio = None
        self.running = False
        
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Pi Detector...")
        
        try:
            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = CameraHandler(
                resolution=tuple(self.config.get("camera.resolution", [640, 480])),
                framerate=self.config.get("camera.framerate", 30)
            )
            
            # Initialize detector
            logger.info("Initializing object detector...")
            self.detector = ObjectDetector(
                model_path=self.config.get("detection.model_path", "models/mobilenet_ssd_v2.tflite"),
                confidence_threshold=self.config.get("detection.confidence_threshold", 0.5)
            )
            
            # Initialize audio system
            if self.config.get("audio.enabled", True):
                logger.info("Initializing audio system...")
                self.audio = AudioOutputSystem(
                    volume=self.config.get("audio.volume", 80)
                )
            
            logger.info("Initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def run(self):
        """Run the main detection loop."""
        if not self.initialize():
            logger.error("Failed to initialize. Exiting.")
            return
        
        self.running = True
        logger.info("Starting detection loop...")
        
        try:
            frame_count = 0
            last_detection = {}
            detection_cooldown = 3  # seconds between announcements
            
            while self.running:
                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Process detections
                current_time = time.time()
                for detection in detections:
                    class_name = detection['class']
                    confidence = detection['confidence']
                    
                    # Check if we should announce this detection
                    if class_name not in last_detection or \
                       (current_time - last_detection[class_name]) > detection_cooldown:
                        
                        logger.info(f"Detected: {class_name} ({confidence:.2f})")
                        
                        # Announce detection via audio
                        if self.audio:
                            message = self._format_detection_message(class_name, confidence)
                            self.audio.speak(message)
                        
                        last_detection[class_name] = current_time
                
                frame_count += 1
                
                # Optional: Add a small delay to prevent CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Stopping...")
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
        finally:
            self.cleanup()
    
    def _format_detection_message(self, class_name: str, confidence: float) -> str:
        """
        Format detection message for audio output.
        
        Args:
            class_name: Detected object class
            confidence: Detection confidence
            
        Returns:
            Formatted message string
        """
        # Customize messages for different classes
        if class_name == "person":
            return "Human detected"
        elif class_name in ["dog", "cat", "bird", "horse", "cow", "sheep"]:
            return f"{class_name.capitalize()} detected"
        else:
            return f"{class_name} detected"
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.running = False
        
        if self.camera:
            self.camera.close()
        
        if self.audio:
            self.audio.close()
        
        logger.info("Cleanup complete. Goodbye!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Raspberry Pi AI Camera Detection System")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/settings.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = PiDetectorApp(config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
