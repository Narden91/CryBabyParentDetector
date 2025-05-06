"""
Audio processing utilities for feature extraction and analysis.
"""
import numpy as np
import librosa
from typing import Dict, Any, Optional

def extract_audio_features(audio_data: np.ndarray, sample_rate: int = 22050) -> Dict[str, Any]:
    """
    Extract comprehensive audio features for analysis.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Audio sample rate
        
    Returns:
        dict: Dictionary of extracted features
    """
    if len(audio_data) == 0:
        return {
            "rms": {"mean": 0, "std": 0},
            "zcr": {"mean": 0, "std": 0},
            "spectral": {
                "centroid": {"mean": 0, "std": 0},
                "rolloff": {"mean": 0, "std": 0},
                "bandwidth": {"mean": 0, "std": 0}
            },
            "mfcc": {"mean": np.zeros(13), "std": np.zeros(13)},
            "amplitude": 0.0
        }
    
    # Calculate audio level (normalized between 0 and 1)
    amplitude = min(np.sqrt(np.mean(audio_data**2)) * 100, 1.0)
    
    # Basic features
    rms = librosa.feature.rms(y=audio_data)[0]
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    
    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
    
    # MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    
    # Organize all features into a structured dictionary
    features = {
        "rms": {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms))
        },
        "zcr": {
            "mean": float(np.mean(zcr)),
            "std": float(np.std(zcr))
        },
        "spectral": {
            "centroid": {
                "mean": float(np.mean(spec_cent)) if len(spec_cent) > 0 else 0,
                "std": float(np.std(spec_cent)) if len(spec_cent) > 0 else 0
            },
            "rolloff": {
                "mean": float(np.mean(spec_rolloff)) if len(spec_rolloff) > 0 else 0,
                "std": float(np.std(spec_rolloff)) if len(spec_rolloff) > 0 else 0
            },
            "bandwidth": {
                "mean": float(np.mean(spec_bw)) if len(spec_bw) > 0 else 0,
                "std": float(np.std(spec_bw)) if len(spec_bw) > 0 else 0
            }
        },
        "mfcc": {
            "mean": np.mean(mfccs, axis=1).tolist(),
            "std": np.std(mfccs, axis=1).tolist()
        },
        "amplitude": float(amplitude)
    }
    
    return features

def preprocess_audio(audio_data: np.ndarray, target_sr: Optional[int] = None) -> np.ndarray:
    """
    Preprocess audio data for analysis.
    
    Args:
        audio_data: Raw audio data as numpy array
        target_sr: Target sample rate (if resampling needed)
        
    Returns:
        np.ndarray: Processed audio data
    """
    # Remove DC offset
    audio_data = librosa.util.normalize(audio_data)
    
    # Apply pre-emphasis filter to enhance high frequencies
    audio_data = librosa.effects.preemphasis(audio_data)
    
    return audio_data

def segment_audio(audio_data: np.ndarray, sample_rate: int = 22050, 
                 segment_length_sec: float = 1.0) -> list:
    """
    Segment audio into fixed-length chunks for analysis.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Audio sample rate
        segment_length_sec: Length of each segment in seconds
        
    Returns:
        list: List of audio segments
    """
    segment_samples = int(sample_rate * segment_length_sec)
    segments = []
    
    # Create segments
    for i in range(0, len(audio_data), segment_samples):
        segment = audio_data[i:i + segment_samples]
        
        # Make sure segment is the right length
        if len(segment) == segment_samples:
            segments.append(segment)
        elif len(segment) > 0:
            # Pad the last segment if needed
            padded_segment = np.zeros(segment_samples)
            padded_segment[:len(segment)] = segment
            segments.append(padded_segment)
    
    return segments