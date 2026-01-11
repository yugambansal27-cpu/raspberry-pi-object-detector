"""
Configuration management module.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for Pi Detector."""
    
    DEFAULT_CONFIG = {
        "camera": {
            "resolution": [640, 480],
            "framerate": 30
        },
        "detection": {
            "confidence_threshold": 0.5,
            "model_path": "models/mobilenet_ssd_v2.tflite"
        },
        "audio": {
            "enabled": True,
            "volume": 80
        },
        "logging": {
            "level": "INFO",
            "file": "pi_detector.log"
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self.DEFAULT_CONFIG.copy()
        
        if self.config_path and self.config_path.exists():
            self.load()
        else:
            logger.info("Using default configuration")
    
    def load(self):
        """Load configuration from file."""
        try:
            logger.info(f"Loading configuration from {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge with defaults
            self._merge_config(self.config, user_config)
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration (uses default if not provided)
        """
        save_path = Path(path) if path else self.config_path
        
        if not save_path:
            logger.error("No path specified for saving configuration")
            return
        
        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "camera.resolution")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "camera.resolution")
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def _merge_config(self, base: dict, override: dict):
        """
        Recursively merge override config into base config.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
