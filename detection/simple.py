"""
Simple level-based detector that uses audio amplitude.
"""
import numpy as np
from typing import Tuple

from .base import BaseDetector

class SimpleLevelDetector(BaseDetector):
    """Simple level-based detector that uses audio amplitude."""
    
    def __init__(self, threshold: float = 0.4):
        """
        Initialize the simple detector.
        
        Args:
            threshold: Audio level threshold for detection (0.0-1.0)
        """
        super().__init__(threshold)
    
    def detect(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect based on audio level.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (is_crying, audio_level)
        """
        if len(audio_data) == 0:
            return False, 0.0
        
        # Calculate RMS (Root Mean Square) level and normalize to 0-1 range
        audio_level = min(np.sqrt(np.mean(audio_data**2)) * 100, 1.0)
        
        return audio_level > self.threshold, audio_level