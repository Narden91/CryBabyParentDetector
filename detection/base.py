"""
Base detector module defining the abstract interface for all detection methods.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

class BaseDetector(ABC):
    """Base class for all baby cry detection methods."""
    
    def __init__(self, threshold: float):
        """
        Initialize the detector with a threshold.
        
        Args:
            threshold: Confidence threshold for detection
        """
        self.threshold = threshold
    
    @abstractmethod
    def detect(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if the audio contains baby crying.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (is_crying, confidence_score)
        """
        pass
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 22050) -> Dict[str, Any]:
        """
        Extract common audio features. Can be overridden by subclasses.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            dict: Dictionary of extracted features
        """
        raise NotImplementedError("Feature extraction not implemented for this detector")