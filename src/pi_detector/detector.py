"""
Object detection module using TensorFlow Lite.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    logging.warning("tflite_runtime not available. Detection will not work.")

import cv2

logger = logging.getLogger(__name__)


# COCO dataset labels (subset relevant to humans and animals)
LABELS = {
    0: "person",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


class ObjectDetector:
    """Object detector using TensorFlow Lite models."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to TFLite model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = LABELS
        
        if TFLITE_AVAILABLE:
            self._load_model()
        else:
            logger.error("TFLite runtime not available. Cannot initialize detector.")
    
    def _load_model(self):
        """Load the TFLite model."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model not found at {self.model_path}. Using dummy detector.")
                return
            
            logger.info(f"Loading model from {self.model_path}")
            
            # Load TFLite model
            self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info("Model loaded successfully")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.interpreter = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Preprocessed image
        """
        if self.interpreter is None:
            return image
        
        # Get input shape
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize image
        resized = cv2.resize(image, (width, height))
        
        # Normalize if required
        input_type = self.input_details[0]['dtype']
        if input_type == np.uint8:
            return resized.astype(np.uint8)
        else:
            # Normalize to [-1, 1] or [0, 1]
            return (resized.astype(np.float32) / 127.5) - 1.0
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run object detection on an image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of detections with class, confidence, and bounding box
        """
        if self.interpreter is None:
            # Return dummy detections for testing without a model
            return self._dummy_detect()
        
        try:
            # Preprocess image
            input_data = self.preprocess_image(image)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get outputs
            # Assuming SSD MobileNet output format:
            # boxes: [1, num_detections, 4]
            # classes: [1, num_detections]
            # scores: [1, num_detections]
            
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Filter detections
            detections = []
            for i in range(len(scores)):
                if scores[i] >= self.confidence_threshold:
                    class_id = int(classes[i])
                    
                    # Only keep humans and animals
                    if class_id in self.labels:
                        detection = {
                            'class': self.labels[class_id],
                            'confidence': float(scores[i]),
                            'bbox': boxes[i].tolist()  # [ymin, xmin, ymax, xmax]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def _dummy_detect(self) -> List[Dict]:
        """
        Return dummy detections for testing.
        Used when model is not available.
        """
        # Return empty list most of the time, occasionally return a test detection
        import random
        if random.random() < 0.1:  # 10% chance
            return [{
                'class': 'person',
                'confidence': 0.85,
                'bbox': [0.2, 0.3, 0.7, 0.6]
            }]
        return []
