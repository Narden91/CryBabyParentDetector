"""
Deep Learning detector using pre-trained Hugging Face models.
"""
import os
import numpy as np
import torch
import torchaudio
import librosa
from typing import Tuple, Optional
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from .base import BaseDetector

class DLDetector(BaseDetector):
    """Deep Learning detector using a pre-trained Hugging Face model for baby cry detection."""
    
    def __init__(self, threshold: float = 0.6, sample_rate: int = 22050):
        """
        Initialize the DL detector with a pre-trained model from Hugging Face.
        
        Args:
            threshold: Confidence threshold for positive detection
            sample_rate: Audio sample rate
        """
        super().__init__(threshold)
        self.sample_rate = sample_rate
        
        # Labels that indicate baby crying
        self.cry_labels = ["baby cry, infant cry"]
        
        # Initialize model and feature extractor
        self._init_model()
    
    def _init_model(self):
        """Initialize the Hugging Face model."""
        try:
            print("Loading pre-trained AST model from Hugging Face...")
            
            # Load feature extractor and model
            self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            self.model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            
            # Get all class labels
            self.id2label = self.model.config.id2label
            
            # Get the indices of baby cry labels
            self.cry_class_indices = [
                idx for idx, label in self.id2label.items() 
                if any(cry_label in label.lower() for cry_label in self.cry_labels)
            ]
            
            if not self.cry_class_indices:
                print("Warning: No baby cry classes found in the model. Using default class index 4 (baby cry in AudioSet).")
                self.cry_class_indices = [4]  # Default index for baby cry in AudioSet
            
            print(f"Model loaded successfully. Baby cry class indices: {self.cry_class_indices}")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio data for the model.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            torch.Tensor: Processed input for the model
        """
        # Resample if needed
        if self.sample_rate != 16000:  # AST models expect 16kHz audio
            audio_data = librosa.resample(audio_data, orig_sr=self.sample_rate, target_sr=16000)
        
        # Ensure audio has enough samples (at least 1 second)
        if len(audio_data) < 16000:
            # Pad if too short
            audio_data = np.pad(audio_data, (0, 16000 - len(audio_data)))
        
        # Convert to torch tensor
        audio_tensor = torch.tensor(audio_data)
        
        # Extract features with AST feature extractor
        inputs = self.feature_extractor(
            audio_tensor, 
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        return inputs
    
    def detect(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect baby crying using the pre-trained model.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (is_crying, confidence_score)
        """
        if len(audio_data) == 0:
            return False, 0.0
        
        try:
            # Preprocess audio
            inputs = self.preprocess_audio(audio_data)
            
            # Move to the same device as model
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            
            # Get probability for baby cry classes (sum of all cry-related classes)
            cry_prob = sum(probs[idx] for idx in self.cry_class_indices)
            
            # Debug info
            if cry_prob > 0.1:  # Only log if there's some probability
                print(f"Baby cry probability: {cry_prob:.4f}")
                
                # Print top 3 classes for debugging
                top_indices = np.argsort(probs)[-3:][::-1]
                for idx in top_indices:
                    print(f"  Class {idx} ({self.id2label[idx]}): {probs[idx]:.4f}")
            
            return cry_prob > self.threshold, float(cry_prob)
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return False, 0.0