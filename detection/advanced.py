"""
Advanced feature-based detector that uses multiple audio characteristics.
"""
import numpy as np
import librosa
from typing import Tuple, Dict, Any

from .base import BaseDetector

class AdvancedFeatureDetector(BaseDetector):
    """Advanced detector using multiple audio features."""
    
    def __init__(self, threshold: float = 0.7, sample_rate: int = 22050):
        """
        Initialize the advanced detector.
        
        Args:
            threshold: Confidence threshold for detection
            sample_rate: Audio sample rate
        """
        super().__init__(threshold)
        self.sample_rate = sample_rate
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int = None) -> Dict[str, Any]:
        """
        Extract audio features for cry detection.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate (overrides instance sample_rate if provided)
            
        Returns:
            dict: Dictionary of extracted features
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        if len(audio_data) == 0:
            return {
                "rms_mean": 0, 
                "zero_crossing_rate_mean": 0, 
                "zero_crossing_rate_std": 0, 
                "centroid_mean": 0,
                "has_high_energy": False,
                "has_high_modulation": False,
                "has_high_pitch": False
            }
        
        # Extract basic features
        rms = librosa.feature.rms(y=audio_data)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        
        # Calculate derived features for baby cry characteristics
        rms_mean = np.mean(rms)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        centroid_mean = np.mean(spec_cent) if len(spec_cent) > 0 else 0
        
        # Detect characteristics of baby cries
        has_high_energy = rms_mean > 0.05  # High energy
        has_high_modulation = zcr_std > 0.1  # High variation in zero-crossing (modulation)
        has_high_pitch = centroid_mean > 2000  # High-pitched sound
        
        return {
            "rms_mean": rms_mean,
            "zero_crossing_rate_mean": zcr_mean,
            "zero_crossing_rate_std": zcr_std,
            "centroid_mean": centroid_mean,
            "has_high_energy": has_high_energy,
            "has_high_modulation": has_high_modulation,
            "has_high_pitch": has_high_pitch
        }
    
    def detect(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect baby crying using multiple features.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (is_crying, confidence_score)
        """
        if len(audio_data) == 0:
            return False, 0.0
        
        # Extract features
        features = self.extract_features(audio_data)
        
        # Calculate confidence score based on feature combination
        confidence = 0.0
        
        if features["has_high_energy"]:
            confidence += 0.3
        if features["has_high_modulation"]:
            confidence += 0.3
        if features["has_high_pitch"]:
            confidence += 0.2
            
        # Bonus for having all three characteristics
        if (features["has_high_energy"] and 
            features["has_high_modulation"] and 
            features["has_high_pitch"]):
            confidence += 0.2
            
        return confidence > self.threshold, confidence