"""
ML-based detector using a pre-trained scikit-learn model.
With feature dimensionality adapter for compatibility with existing models.
"""
import os
import numpy as np
import librosa
import joblib
import warnings
from typing import Tuple, Optional
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector

class MLDetector(BaseDetector):
    """ML-based detector using a pre-trained scikit-learn model (SVM)."""
    
    def __init__(self, model_path: str, threshold: float = 0.6, sample_rate: int = 22050, 
                 create_dummy_model: bool = True):
        """
        Initialize the ML detector.
        
        Args:
            model_path: Path to the joblib model file
            threshold: Confidence threshold for positive detection
            sample_rate: Audio sample rate
            create_dummy_model: Whether to create a dummy model if model_path not found
        """
        super().__init__(threshold)
        self.sample_rate = sample_rate
        self.model_path = model_path
        self.expected_feature_count = None
        
        # Try to load the model, or create a dummy model if requested
        self.model = self._load_model(model_path, create_dummy_model)
        self.model_type = type(self.model).__name__
    
    def _load_model(self, model_path: str, create_dummy: bool) -> object:
        """
        Load model from path or create a dummy model.
        
        Args:
            model_path: Path to model file
            create_dummy: Whether to create a dummy model if path not found
            
        Returns:
            object: The loaded or created model
        """
        try:
            model = joblib.load(model_path)
            print(f"Loaded ML model: {type(model).__name__} from {model_path}")
            
            # Try to determine expected feature count
            if hasattr(model, 'n_features_in_'):
                self.expected_feature_count = model.n_features_in_
            elif hasattr(model, 'steps'):
                # For Pipeline objects
                for _, step in reversed(model.steps):
                    if hasattr(step, 'n_features_in_'):
                        self.expected_feature_count = step.n_features_in_
                        break
            
            print(f"Model expects {self.expected_feature_count} features")
            return model
            
        except (FileNotFoundError, Exception) as e:
            print(f"Could not load model from {model_path}: {e}")
            
            if create_dummy:
                print("Creating dummy SVM model for baby cry detection")
                # Create a simple dummy model that works with our features
                dummy_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(kernel='rbf', probability=True))
                ])
                
                # Get feature count from our extractor with a sample
                sample_features = self.extract_features(np.random.random(self.sample_rate))
                n_features = sample_features.shape[1]
                self.expected_feature_count = n_features
                
                # Fit with dummy data to initialize
                X = np.random.random((10, n_features))
                y = np.random.choice([0, 1], 10)
                dummy_model.fit(X, y)
                
                # Save for future use
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(dummy_model, model_path)
                
                print(f"Dummy model created with {n_features} features")
                return dummy_model
            else:
                raise
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract features for ML model prediction.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            np.ndarray: Feature vector for ML model
        """
        if len(audio_data) == 0:
            # Return empty feature vector with appropriate dimensions
            feature_count = 41  # Default feature count from our extraction
            return np.zeros((1, feature_count))
        
        # Apply a window if audio is very short
        if len(audio_data) < 1024:
            audio_data = np.pad(audio_data, (0, 1024 - len(audio_data)))
        
        # Extract MFCC features (commonly used for audio classification)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Extract other useful features
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        rms = librosa.feature.rms(y=audio_data)[0]
        
        # For very short audio, handle spectral features carefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spec_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            spec_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
        
        spec_contrast_mean = np.mean(spec_contrast, axis=1) if len(spec_contrast) > 0 else np.zeros(6)
        
        # Combine all features
        feature_vector = np.concatenate([
            mfcc_mean,           # 13 features
            mfcc_std,            # 13 features
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)],
            [np.mean(spec_cent), np.std(spec_cent)],
            [np.mean(spec_rolloff), np.std(spec_rolloff)],
            spec_contrast_mean   # 6 features
        ])
        
        return feature_vector.reshape(1, -1)  # Reshape for sklearn prediction
    
    def adapt_feature_dimensionality(self, features: np.ndarray) -> np.ndarray:
        """
        Adapt features to match the expected dimensionality of the model.
        
        This is critical for making the detector work with pre-trained models
        that expect a different number of features.
        
        Args:
            features: Extracted feature vector
            
        Returns:
            np.ndarray: Feature vector adapted to the model's expected dimensions
        """
        if self.expected_feature_count is None:
            # If we don't know expected count, just return as is
            return features
            
        current_count = features.shape[1]
        
        if current_count == self.expected_feature_count:
            # No adaptation needed
            return features
            
        # Need to adapt - create a new feature array
        adapted = np.zeros((1, self.expected_feature_count))
        
        if current_count < self.expected_feature_count:
            # Our features are fewer than expected - copy what we have
            adapted[0, :current_count] = features[0, :]
            # Fill the rest with zeros (or could use means, etc.)
        else:
            # Our features are more than expected - truncate
            adapted[0, :] = features[0, :self.expected_feature_count]
            
        return adapted
    
    def detect(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect using ML model.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (is_crying, confidence_score)
        """
        if len(audio_data) == 0:
            return False, 0.0
        
        # For very short audio, consider it not crying
        if len(audio_data) < 100:
            return False, 0.0
            
        try:
            # Extract features
            features = self.extract_features(audio_data)
            
            # Adapt features to match model expectations
            features = self.adapt_feature_dimensionality(features)
            
            # For models that provide probability estimates
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)
                if proba.shape[1] > 1:  # Binary classification
                    confidence = proba[0][1]  # Probability of positive class
                else:
                    confidence = proba[0][0]
            else:
                # For models without probability support
                prediction = self.model.predict(features)[0]
                confidence = float(prediction)  # 0 or 1
        except Exception as e:
            print(f"Error during ML prediction: {e}")
            
            # Fall back to the advanced detector method 
            # (calculating a confidence based on audio characteristics)
            energy = np.mean(audio_data**2) * 100
            modulation = np.std(librosa.feature.zero_crossing_rate(audio_data)[0])
            
            # Simple fallback detection using basic audio characteristics
            energy_high = energy > 0.05
            modulation_high = modulation > 0.1
            
            confidence = 0.0
            if energy_high:
                confidence += 0.4
            if modulation_high:
                confidence += 0.4
                
            print(f"Using fallback detection method, confidence: {confidence:.2f}")
            
        return confidence > self.threshold, confidence