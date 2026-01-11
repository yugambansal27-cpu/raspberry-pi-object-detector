"""
Audio output system for announcing detections.
"""

import logging
from typing import Optional
import threading
import queue

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available. Audio output disabled.")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available.")

logger = logging.getLogger(__name__)


class AudioOutputSystem:
    """Handles audio output for detection announcements."""
    
    def __init__(self, volume: int = 80):
        """
        Initialize audio output system.
        
        Args:
            volume: Volume level (0-100)
        """
        self.volume = min(100, max(0, volume))
        self.engine = None
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        self._initialize_engine()
        self._start_worker()
    
    def _initialize_engine(self):
        """Initialize text-to-speech engine."""
        if not PYTTSX3_AVAILABLE:
            logger.warning("Audio output not available (pyttsx3 missing)")
            return
        
        try:
            logger.info("Initializing TTS engine...")
            self.engine = pyttsx3.init()
            
            # Configure engine
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', self.volume / 100.0)
            
            # Try to set a clear voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer English voice
                for voice in voices:
                    if 'english' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            logger.info("TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def _start_worker(self):
        """Start background worker thread for audio playback."""
        if not self.engine:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audio worker thread started")
    
    def _audio_worker(self):
        """Background worker for processing audio queue."""
        while self.running:
            try:
                # Get message from queue with timeout
                message = self.speech_queue.get(timeout=1.0)
                
                if message is None:
                    # Poison pill to stop worker
                    break
                
                # Speak the message
                self._speak_sync(message)
                
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio worker: {e}")
    
    def _speak_sync(self, text: str):
        """
        Synchronously speak text.
        
        Args:
            text: Text to speak
        """
        if not self.engine:
            logger.debug(f"Would speak: {text}")
            return
        
        try:
            logger.debug(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
    
    def speak(self, text: str):
        """
        Asynchronously speak text.
        
        Args:
            text: Text to speak
        """
        if not self.engine:
            logger.info(f"Audio: {text}")
            return
        
        try:
            # Add to queue
            self.speech_queue.put(text)
            logger.debug(f"Queued for speech: {text}")
        except Exception as e:
            logger.error(f"Error queueing speech: {e}")
    
    def play_sound(self, sound_file: str):
        """
        Play a sound file.
        
        Args:
            sound_file: Path to sound file
        """
        if not PYGAME_AVAILABLE:
            logger.warning("Cannot play sound file (pygame not available)")
            return
        
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            logger.info(f"Playing sound: {sound_file}")
        except Exception as e:
            logger.error(f"Error playing sound: {e}")
    
    def set_volume(self, volume: int):
        """
        Set volume level.
        
        Args:
            volume: Volume level (0-100)
        """
        self.volume = min(100, max(0, volume))
        
        if self.engine:
            try:
                self.engine.setProperty('volume', self.volume / 100.0)
                logger.info(f"Volume set to {self.volume}%")
            except Exception as e:
                logger.error(f"Error setting volume: {e}")
    
    def close(self):
        """Clean up audio resources."""
        logger.info("Closing audio system...")
        
        # Stop worker thread
        self.running = False
        if self.worker_thread:
            self.speech_queue.put(None)  # Poison pill
            self.worker_thread.join(timeout=2.0)
        
        # Clean up engine
        if self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"Error stopping TTS engine: {e}")
        
        logger.info("Audio system closed")
