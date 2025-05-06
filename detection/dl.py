"""
Deep Learning detector using pre-trained Hugging Face models specifically for baby cry detection.
"""
import os
import numpy as np
import torch
import torchaudio
import librosa
import time
import logging
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BabyCryDL")

from .base import BaseDetector

class DLDetector(BaseDetector):
    """Deep Learning detector using a pre-trained Hugging Face model optimized for baby cry detection."""
    
    # Lista dei modelli supportati in ordine di preferenza
    SUPPORTED_MODELS = {
        "MIT/ast-finetuned-audioset-10-10-0.4593": {
            "type": "ast",
            "cry_labels": ["baby cry", "infant cry", "crying baby"],
            "default_cry_indices": [4]  # Indice di default per AudioSet
        },
        "microsoft/wavlm-base-plus": {
            "type": "wavlm",
            "adapter_hidden_size": 768
        }
    }
    
    def __init__(self, threshold: float = 0.6, sample_rate: int = 22050, 
                 model_name: str = None, cache_dir: str = None,
                 use_custom_processor: bool = True):
        """
        Initialize the DL detector with a pre-trained model from Hugging Face.
        
        Args:
            threshold: Confidence threshold for positive detection
            sample_rate: Audio sample rate
            model_name: Specific model name to use (if None, will try models in order)
            cache_dir: Directory to cache downloaded models
            use_custom_processor: Whether to use custom audio processing
        """
        super().__init__(threshold)
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "babycry")
        self.use_custom_processor = use_custom_processor
        
        # Assicuriamoci che la cache directory esista
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Labels che indicano il pianto di un neonato (verranno aggiornati in base al modello)
        self.cry_labels = ["baby cry", "infant cry", "crying baby", "child crying", "crying, sobbing"]
        
        # Informazioni sul modello
        self.model_type = None
        self.model_info = {}
        self.cry_class_indices = []
        self.model_loaded = False
        self.last_load_attempt = 0
        self.load_retry_interval = 3600  # 1 ora tra i tentativi di caricamento
        
        # Statistiche
        self.stats = {
            "detection_count": 0,
            "false_positive_count": 0,
            "processing_times": []
        }
        
        # Inizializza il modello
        try:
            self._init_model()
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            logger.info("The detector will retry loading the model later.")
    
    def _init_model(self):
        """
        Initialize the Hugging Face model with proper error handling and fallbacks.
        """
        # Registra il tentativo di caricamento
        self.last_load_attempt = time.time()
        
        # Se è stato specificato un modello, usa quello
        models_to_try = [self.model_name] if self.model_name else self.SUPPORTED_MODELS.keys()
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                
                if model_name in self.SUPPORTED_MODELS:
                    self.model_info = self.SUPPORTED_MODELS[model_name]
                    self.model_type = self.model_info["type"]
                else:
                    # Modello non conosciuto, supponiamo sia AST
                    self.model_info = {"type": "ast"}
                    self.model_type = "ast"
                
                # Carica il modello in base al tipo
                if self.model_type == "ast":
                    self._load_ast_model(model_name)
                elif self.model_type == "wavlm":
                    self._load_wavlm_model(model_name)
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")
                
                self.model_loaded = True
                self.model_name = model_name
                logger.info(f"Successfully loaded model: {model_name}")
                break
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                continue
        
        if not self.model_loaded:
            logger.warning("Failed to load any model, will use fallback detection method")
    
    def _load_ast_model(self, model_name):
        """
        Load an Audio Spectrogram Transformer model.
        """
        try:
            # Import qui per evitare import errors se transformers non è disponibile
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            
            # Carica il feature extractor e il modello
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(
                model_name, cache_dir=self.cache_dir
            )
            self.model = ASTForAudioClassification.from_pretrained(
                model_name, cache_dir=self.cache_dir
            )
            
            # Ottieni tutte le etichette delle classi
            self.id2label = self.model.config.id2label
            
            # Ottieni gli indici delle classi di pianto
            if "cry_labels" in self.model_info:
                self.cry_labels = self.model_info["cry_labels"]
            
            self.cry_class_indices = self._find_cry_class_indices()
            
            if not self.cry_class_indices and "default_cry_indices" in self.model_info:
                logger.info(f"No baby cry classes found in model. Using default indices: {self.model_info['default_cry_indices']}")
                self.cry_class_indices = self.model_info["default_cry_indices"]
            
            if not self.cry_class_indices:
                logger.warning("No baby cry classes found in the model and no defaults available.")
                # Use a fallback approach
                self.cry_class_indices = [4]  # Default index for baby cry in AudioSet
            
            logger.info(f"Model loaded successfully. Baby cry class indices: {self.cry_class_indices}")
            
            # Imposta device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()  # Imposta in modalità valutazione
            
        except Exception as e:
            logger.error(f"Error loading AST model: {e}")
            raise
    
    def _load_wavlm_model(self, model_name):
        """
        Load a WavLM model with a custom classification head for baby cry detection.
        """
        try:
            # Import qui per evitare import errors se transformers non è disponibile
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel
            import torch.nn as nn
            
            # Carica il feature extractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_name, cache_dir=self.cache_dir
            )
            
            # Carica il modello base
            wavlm_model = WavLMModel.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Definisci un classificatore personalizzato
            hidden_size = self.model_info.get("adapter_hidden_size", 768)
            
            class WavLMBabyCryClassifier(nn.Module):
                def __init__(self, wavlm_model, hidden_size):
                    super().__init__()
                    self.wavlm = wavlm_model
                    self.classifier = nn.Sequential(
                        nn.Linear(hidden_size, 256),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)  # 2 classi: pianto e non pianto
                    )
                
                def forward(self, input_values, attention_mask=None):
                    outputs = self.wavlm(input_values, attention_mask=attention_mask)
                    # Pooling: usa la media dei hidden states
                    pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
                    logits = self.classifier(pooled_output)
                    return logits
            
            # Crea il modello combinato
            self.model = WavLMBabyCryClassifier(wavlm_model, hidden_size)
            
            # Imposta le label
            self.id2label = {0: "no_cry", 1: "baby_cry"}
            
            # Imposta gli indici delle classi di pianto
            self.cry_class_indices = [1]  # Indice 1 = baby_cry
            
            # Carica i pesi del classificatore custom se disponibili
            custom_weights_path = os.path.join(self.cache_dir, "wavlm_babycry_classifier.pt")
            if os.path.exists(custom_weights_path):
                logger.info(f"Loading custom classifier weights from {custom_weights_path}")
                self.model.classifier.load_state_dict(torch.load(custom_weights_path))
            else:
                logger.warning("No custom classifier weights found, using untrained classifier")
                # In un'implementazione reale, qui dovresti caricare pesi pre-addestrati o
                # addestrare il classificatore con dati di pianto di neonati
            
            # Imposta device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()  # Imposta in modalità valutazione
            
        except Exception as e:
            logger.error(f"Error loading WavLM model: {e}")
            raise
    
    def _find_cry_class_indices(self) -> List[int]:
        """
        Find indices of baby cry related classes in the model.
        
        Returns:
            list: List of class indices
        """
        cry_indices = []
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(cry_label in label_lower for cry_label in self.cry_labels):
                cry_indices.append(idx)
                logger.info(f"Found baby cry class: {label} (index {idx})")
        
        return cry_indices
    
    def _ensure_model_loaded(self) -> bool:
        """
        Ensure the model is loaded, attempt to load it if not.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        if self.model_loaded:
            return True
        
        # Check if we should retry loading
        current_time = time.time()
        if current_time - self.last_load_attempt > self.load_retry_interval:
            try:
                self._init_model()
                return self.model_loaded
            except Exception as e:
                logger.error(f"Failed to load model during retry: {e}")
                self.last_load_attempt = current_time
        
        return False
    
    def preprocess_audio(self, audio_data: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Preprocess audio data for the model with advanced normalization and augmentation.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            dict: Processed inputs for the model
        """
        # Preliminary check
        if len(audio_data) == 0:
            # Create empty input tensor with appropriate shape
            return {"input_values": torch.zeros((1, 16000))}
        
        try:
            # Resample if needed - most HF models expect 16kHz audio
            target_sr = 16000
            if self.sample_rate != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=self.sample_rate, target_sr=target_sr)
            
            # Normalize audio (robust normalization)
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Apply preprocessing based on model type
            if not self.model_loaded or not self.use_custom_processor:
                # Simplified preprocessing when model isn't loaded
                # Ensure audio has enough samples (at least 1 second)
                if len(audio_data) < target_sr:
                    # Pad if too short
                    audio_data = np.pad(audio_data, (0, target_sr - len(audio_data)))
                # Trim if too long
                elif len(audio_data) > 5*target_sr:  # Max 5 seconds
                    audio_data = audio_data[:5*target_sr]
                
                # Convert to tensor
                audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                return {"input_values": audio_tensor}
            
            # Model-specific preprocessing
            if self.model_type == "ast":
                # Convert to torch tensor
                audio_tensor = torch.tensor(audio_data)
                
                # Extract features with AST feature extractor
                inputs = self.feature_extractor(
                    audio_tensor, 
                    sampling_rate=target_sr,
                    return_tensors="pt"
                )
                return inputs
                
            elif self.model_type == "wavlm":
                # Pad or trim to 3 seconds (reasonable length for cry detection)
                target_length = 3 * target_sr
                if len(audio_data) < target_length:
                    audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
                else:
                    audio_data = audio_data[:target_length]
                
                # Process with feature extractor
                inputs = self.feature_extractor(
                    audio_data, 
                    sampling_rate=target_sr,
                    return_tensors="pt"
                )
                return inputs
                
            else:
                # Fallback for unknown model types
                audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                return {"input_values": audio_tensor}
                
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            # Return empty tensor as fallback
            return {"input_values": torch.zeros((1, target_sr))}
    
    def detect(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect baby crying using the pre-trained model with fallback mechanisms.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (is_crying, confidence_score)
        """
        start_time = time.time()
        
        # Initial check
        if len(audio_data) == 0:
            return False, 0.0
        
        # For very short audio, consider it not crying
        if len(audio_data) < 100:
            return False, 0.0
        
        # Check if model is loaded
        model_available = self._ensure_model_loaded()
        
        try:
            if model_available:
                # Use deep learning model
                return self._detect_with_model(audio_data)
            else:
                # Use fallback method
                return self._detect_fallback(audio_data)
                
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            # Use fallback method
            return self._detect_fallback(audio_data)
        finally:
            # Update processing time stats
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            # Keep only the last 100 processing times
            if len(self.stats["processing_times"]) > 100:
                self.stats["processing_times"].pop(0)
    
    def _detect_with_model(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect crying using the loaded deep learning model.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (is_crying, confidence_score)
        """
        # Preprocess audio
        inputs = self.preprocess_audio(audio_data)
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            if self.model_type == "ast":
                outputs = self.model(**inputs)
                # Get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                
                # Get probability for baby cry classes (sum of all cry-related classes)
                cry_prob = sum(probs[idx] for idx in self.cry_class_indices)
                
                # Debug info
                if cry_prob > 0.1:  # Solo log significativi
                    logger.info(f"Baby cry probability: {cry_prob:.4f}")
                    
                    # Print top 3 classes for debugging
                    top_indices = np.argsort(probs)[-3:][::-1]
                    for idx in top_indices:
                        logger.info(f"  Class {idx} ({self.id2label[idx]}): {probs[idx]:.4f}")
                
            elif self.model_type == "wavlm":
                logits = self.model(**inputs)
                probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
                cry_prob = float(probs[1])  # 1 = baby_cry class
                
                if cry_prob > 0.1:
                    logger.info(f"Baby cry probability: {cry_prob:.4f}")
            else:
                # Fallback for unknown model type
                return self._detect_fallback(audio_data)
            
            # Update stats
            is_crying = cry_prob > self.threshold
            if is_crying:
                self.stats["detection_count"] += 1
            
            return is_crying, float(cry_prob)
            
    def _detect_fallback(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Fallback detection method when the model is not available.
        Uses audio characteristics to determine if it's a baby cry.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            tuple: (is_crying, confidence_score)
        """
        # Calculate energy (overall loudness)
        energy = np.mean(audio_data**2) * 100
        
        # Calculate zero-crossing rate (related to pitch)
        try:
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
        except Exception:
            zcr_mean = 0
            zcr_std = 0
        
        # Calculate spectral centroid (related to brightness/sharpness)
        try:
            spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            centroid_mean = np.mean(spec_cent) if len(spec_cent) > 0 else 0
        except Exception:
            centroid_mean = 0
        
        # Features typical of baby cries
        energy_high = energy > 0.05
        modulation_high = zcr_std > 0.1
        pitch_high = centroid_mean > 2000
        
        # Calculate confidence score
        confidence = 0.0
        if energy_high:
            confidence += 0.3
        if modulation_high:
            confidence += 0.3
        if pitch_high:
            confidence += 0.2
        if energy_high and modulation_high and pitch_high:
            confidence += 0.2
        
        logger.info(f"Fallback detection: energy={energy:.2f}, zcr_std={zcr_std:.2f}, centroid={centroid_mean:.0f}, confidence={confidence:.2f}")
        
        return confidence > self.threshold, confidence
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            dict: Model information
        """
        avg_processing_time = 0
        if self.stats["processing_times"]:
            avg_processing_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
        
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_loaded": self.model_loaded,
            "device": str(self.device) if hasattr(self, 'device') else "cpu",
            "cry_class_indices": self.cry_class_indices,
            "threshold": self.threshold,
            "detection_count": self.stats["detection_count"],
            "avg_processing_time": avg_processing_time
        }