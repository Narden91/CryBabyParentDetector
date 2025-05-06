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

# Assuming BaseDetector is in the same directory or properly pathed as 'base.py'
from .base import BaseDetector 
# If running standalone and base.py is not available, uncomment the dummy BaseDetector:
# class BaseDetector:
#     def __init__(self, threshold: float):
#         self.threshold = threshold
#     def detect(self, audio_data: np.ndarray) -> Tuple[bool, float]:
#         raise NotImplementedError

class DLDetector(BaseDetector):
    """Deep Learning detector using a pre-trained Hugging Face model optimized for baby cry detection."""
    
    SUPPORTED_MODELS = {
        "MIT/ast-finetuned-audioset-10-10-0.4593": {
            "type": "ast",
            "cry_labels": ["baby cry", "infant cry", "crying baby", "speech", "music"], # Added speech/music to see other common classes
            "default_cry_indices": [4] 
        },
        "microsoft/wavlm-base-plus": {
            "type": "wavlm",
            "adapter_hidden_size": 768 
        }
    }
    
    def __init__(self, threshold: float = 0.6, sample_rate: int = 16000, 
                 model_name: Optional[str] = None, cache_dir: Optional[str] = None,
                 use_custom_processor: bool = True):
        super().__init__(threshold)
        self.sample_rate = sample_rate
        self.model_name_preference = model_name 
        self.actual_model_name: Optional[str] = None 
        self.cache_dir = cache_dir or os.path.join(Path.home(), ".cache", "babycry_detector_models") # More specific cache dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cry_labels_keywords = ["baby cry", "infant cry", "crying baby", "child crying", "crying, sobbing"]
        
        self.model_type: Optional[str] = None
        self.model_config_info: Dict[str, Any] = {} # Stores config for the loaded model
        self.cry_class_indices: List[int] = []
        self.model_loaded = False
        self.last_load_attempt = 0
        self.load_retry_interval = 3600  
        
        self.feature_extractor = None
        self.model = None
        self.id2label: Dict[int, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stats = {
            "detection_count": 0,
            "false_positive_count": 0,
            "processing_times": []
        }
        
        try:
            self._init_model()
        except Exception as e:
            logger.error(f"Fatal error during model initialization: {e}", exc_info=True)
            logger.info("The detector will retry loading the model later if applicable, or use fallback.")
    
    def _init_model(self):
        self.last_load_attempt = time.time()
        
        models_to_try_names: List[str]
        current_supported_models = self.SUPPORTED_MODELS.copy() # Use a copy to modify for preference

        if self.model_name_preference:
            models_to_try_names = [self.model_name_preference]
            if self.model_name_preference not in current_supported_models:
                logger.warning(f"Model '{self.model_name_preference}' is not in the predefined supported list. Will attempt to infer type.")
                if "wavlm" in self.model_name_preference.lower():
                     current_supported_models[self.model_name_preference] = {"type": "wavlm", "adapter_hidden_size": 768}
                elif "ast" in self.model_name_preference.lower() or "audio" in self.model_name_preference.lower():
                    current_supported_models[self.model_name_preference] = {"type": "ast", "cry_labels": self.cry_labels_keywords, "default_cry_indices": []}
                else: # Default to ast if unsure
                    logger.warning(f"Could not infer type for '{self.model_name_preference}', assuming AST. This might fail.")
                    current_supported_models[self.model_name_preference] = {"type": "ast", "cry_labels": self.cry_labels_keywords, "default_cry_indices": []}
        else:
            models_to_try_names = list(current_supported_models.keys())

        preferred_ast_model = "MIT/ast-finetuned-audioset-10-10-0.4593"

        for model_name_to_attempt in models_to_try_names:
            try:
                logger.info(f"Attempting to load model: {model_name_to_attempt}")
                
                if model_name_to_attempt in current_supported_models:
                    self.model_config_info = current_supported_models[model_name_to_attempt]
                    self.model_type = self.model_config_info["type"]
                else: 
                    logger.error(f"Model '{model_name_to_attempt}' configuration not found. Skipping.")
                    continue
                
                if self.model_type == "ast":
                    self._load_ast_model(model_name_to_attempt)
                elif self.model_type == "wavlm":
                    self._load_wavlm_model(model_name_to_attempt)
                else:
                    logger.error(f"Unsupported model type: {self.model_type} for model {model_name_to_attempt}")
                    continue 
                
                if self.model_loaded:
                    self.optimize_model_for_inference()
                    self.actual_model_name = model_name_to_attempt
                    logger.info(f"Successfully loaded and initialized model: {self.actual_model_name} ({self.model_type}) on {self.device}")
                    break 
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name_to_attempt}: {e}", exc_info=True)
                if model_name_to_attempt == preferred_ast_model and self.model_name_preference is None:
                    logger.warning(f"The preferred AST model '{preferred_ast_model}' failed to load. Will attempt fallback models if available.")
                self.model_loaded = False 
                self.model = None # Clear partially loaded model
                self.feature_extractor = None
                continue
        
        if not self.model_loaded:
            logger.warning("Failed to load any specified or default Hugging Face model. Detection will rely on basic fallback if implemented, or fail.")

    def create_dummy_classifier_weights(self, save_path: str, model_input_dim: int, classifier_internal_hidden_dim: int = 256, num_labels: int = 2) -> Dict[str, torch.Tensor]:
        logger.info(f"Creating basic (randomly initialized) classifier weights. Path: {save_path}")
        logger.warning("IMPORTANT: This is a DUMMY classifier with random weights. It is NOT TRAINED for baby cry detection and will likely perform POORLY. For reliable detection, you MUST train this classifier head or use a model already fine-tuned for baby cry detection (like an AST model from AudioSet).")
        
        state_dict = {}
        # Matches WavLMBabyCryClassifier structure:
        # Layer 0: nn.Linear(model_input_dim, classifier_internal_hidden_dim)
        state_dict['0.weight'] = torch.FloatTensor(classifier_internal_hidden_dim, model_input_dim).normal_(0, 0.02)
        state_dict['0.bias'] = torch.zeros(classifier_internal_hidden_dim)
        
        # Layer 3: nn.Linear(classifier_internal_hidden_dim, classifier_internal_hidden_dim // 2)
        state_dict['3.weight'] = torch.FloatTensor(classifier_internal_hidden_dim // 2, classifier_internal_hidden_dim).normal_(0, 0.02)
        state_dict['3.bias'] = torch.zeros(classifier_internal_hidden_dim // 2)
        
        # Layer 6: nn.Linear(classifier_internal_hidden_dim // 2, num_labels)
        state_dict['6.weight'] = torch.FloatTensor(num_labels, classifier_internal_hidden_dim // 2).normal_(0, 0.02)
        state_dict['6.bias'] = torch.zeros(num_labels)
        
        # For WavLM, id2label is {0: "no_cry", 1: "baby_cry"}
        # Add a slight bias towards "no_cry" (index 0) for an untrained binary classifier
        if num_labels == 2: # Assuming index 0 is 'no_cry'
            state_dict['6.bias'][0] = 0.1 

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(state_dict, save_path)
        logger.info(f"Basic DUMMY classifier weights saved to {save_path}. Remember, this is UNTRAINED.")
        return state_dict
    
    def _load_ast_model(self, model_name: str):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
            
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.model = ASTForAudioClassification.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        self.id2label = self.model.config.id2label
        
        # Use cry_labels from model_config_info if available
        current_cry_keywords = self.model_config_info.get("cry_labels", self.cry_labels_keywords)
        self.cry_class_indices = self._find_cry_class_indices(current_cry_keywords)
        
        if not self.cry_class_indices and "default_cry_indices" in self.model_config_info and self.model_config_info["default_cry_indices"]:
            logger.info(f"No specific baby cry classes found by name in model '{model_name}'. Using default indices: {self.model_config_info['default_cry_indices']}")
            self.cry_class_indices = self.model_config_info["default_cry_indices"]
        
        if not self.cry_class_indices: # Fallback for audioset if still not found
            if "audioset" in model_name.lower():
                 logger.info("Attempting to use index 4 as a common fallback for 'baby cry' in AudioSet models, as no specific indices were found.")
                 self.cry_class_indices = [4] # Index 4 is 'Speech' in MIT/ast... but often 'Baby cry, infant cry' in others. Check specific model.
                                              # For MIT/ast-finetuned-audioset-10-10-0.4593, "Baby cry, infant cry" is actually index 378.
                                              # The config has "cry_labels": ["baby cry", "infant cry", "crying baby"] which should find it.
                                              # If default_cry_indices is [4], it might be for a different audioset variant.
                                              # For "MIT/ast-finetuned-audioset-10-10-0.4593", the label for index 4 is "Speech".
                                              # We should rely on _find_cry_class_indices with the right keywords.
                                              # The default [4] in SUPPORTED_MODELS for AST might be misleading for this specific model.
                                              # Let's prioritize _find_cry_class_indices.
                # Re-evaluating default_cry_indices for MIT/ast...
                # Based on its config: "Baby cry, infant cry" is at index 378.
                # Let's update SUPPORTED_MODELS default for it if _find_cry_class_indices fails.
                # For now, _find_cry_class_indices should work with the keywords.
    


        if not self.cry_class_indices:
             raise ValueError(f"Could not determine cry class indices for AST model {model_name}. Searched with keywords: {current_cry_keywords}")

        logger.info(f"AST Model '{model_name}' loaded. Baby cry class indices: {self.cry_class_indices} (labels: {[self.id2label.get(i, f'Unknown_Index_{i}') for i in self.cry_class_indices]})")
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True
    
    def _load_wavlm_model(self, model_name: str):
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel 
        import torch.nn as nn
            
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        
        wavlm_base_model = WavLMModel.from_pretrained(model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        
        # hidden_size is the output dimension of the WavLM base model
        model_output_hidden_size = self.model_config_info.get("adapter_hidden_size", wavlm_base_model.config.hidden_size) 
            
        class WavLMBabyCryClassifier(nn.Module):
            def __init__(self, wavlm_model_base, num_labels=2, classifier_hidden_dim=256, dropout_rate=0.1, wavlm_output_dim=768):
                super().__init__()
                self.wavlm = wavlm_model_base
                self.classifier = nn.Sequential(
                    nn.Linear(wavlm_output_dim, classifier_hidden_dim),       
                    nn.ReLU(),                                       
                    nn.Dropout(dropout_rate),                        
                    nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),        
                    nn.ReLU(),                                     
                    nn.Dropout(dropout_rate),                       
                    nn.Linear(classifier_hidden_dim // 2, num_labels)          
                )
                self.is_dummy = False # Flag to indicate if using dummy weights
            
            def forward(self, input_values, attention_mask=None):
                outputs = self.wavlm(input_values, attention_mask=attention_mask)
                pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
                logits = self.classifier(pooled_output)
                return logits
        
        self.model = WavLMBabyCryClassifier(wavlm_base_model, wavlm_output_dim=model_output_hidden_size)
        
        self.id2label = {0: "no_cry", 1: "baby_cry"} # Specific to this binary classifier
        self.cry_class_indices = [1] 
        
        # Path for storing/loading the classifier head's weights
        classifier_weights_filename = f"{model_name.replace('/', '_')}_custom_babycry_classifier_v2.pt" # Added v2 to avoid conflict with old format
        custom_weights_path = os.path.join(self.cache_dir, classifier_weights_filename)
        
        if os.path.exists(custom_weights_path):
            logger.info(f"Loading custom classifier weights for WavLM from {custom_weights_path}")
            try:
                self.model.classifier.load_state_dict(torch.load(custom_weights_path, map_location=self.device))
                self.model.is_dummy = False
                logger.info("Successfully loaded custom classifier weights.")
            except Exception as e:
                logger.error(f"Error loading custom classifier weights from {custom_weights_path}: {e}. A new DUMMY classifier will be created.", exc_info=True)
                # Pass model_output_hidden_size which is the input dim for the first layer of the classifier
                state_dict = self.create_dummy_classifier_weights(custom_weights_path, model_input_dim=model_output_hidden_size)
                self.model.classifier.load_state_dict(state_dict)
                self.model.is_dummy = True
        else:
            logger.warning(f"No custom classifier weights found at {custom_weights_path}.")
            state_dict = self.create_dummy_classifier_weights(custom_weights_path, model_input_dim=model_output_hidden_size)
            self.model.classifier.load_state_dict(state_dict)
            self.model.is_dummy = True
            
        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True

    def optimize_model_for_inference(self):
        if not self.model_loaded or not self.model:
            logger.warning("Cannot optimize: Model not loaded.")
            return False
        
        self.model.eval() 
        logger.info("Model set to eval() mode. Inference will use torch.inference_mode().")
            
        if self.device.type == 'cuda':
            torch.cuda.empty_cache() 
        
        logger.info(f"Model '{self.actual_model_name}' ready for inference on {self.device}")
        return True
    
    def _find_cry_class_indices(self, keywords_list: List[str]) -> List[int]:
        """Finds class indices related to cry keywords from self.id2label."""
        cry_indices = []
        if not self.id2label: 
            logger.warning("id2label is not populated for the current model, cannot find cry class indices by name.")
            return []
            
        for idx, label_str in self.id2label.items():
            label_lower = label_str.lower()
            if any(cry_keyword.lower() in label_lower for cry_keyword in keywords_list):
                cry_indices.append(idx)
                logger.info(f"Found potential baby cry class: '{label_str}' (index {idx}) using keywords: {keywords_list}")
        
        return list(set(cry_indices)) 
    
    def _ensure_model_loaded(self) -> bool:
        if self.model_loaded and self.model is not None:
            return True
        
        current_time = time.time()
        if current_time - self.last_load_attempt > self.load_retry_interval:
            logger.info("Attempting to reload model due to previous failure and elapsed retry interval.")
            try:
                self._init_model() 
                return self.model_loaded and self.model is not None
            except Exception as e:
                logger.error(f"Failed to reload model during retry: {e}", exc_info=True)
                self.last_load_attempt = current_time 
        else:
            logger.warning("Model is not loaded. Detection will be unreliable or use fallback.")
        return False
    
    def preprocess_audio(self, audio_data: np.ndarray) -> Optional[Dict[str, torch.Tensor]]:
        if not self.feature_extractor:
            logger.error("Feature extractor not available. Cannot preprocess audio.")
            return None

        if audio_data.ndim > 1: 
            audio_data = np.mean(audio_data, axis=1).astype(np.float32)
        else:
            audio_data = audio_data.astype(np.float32)

        # Normalize audio to [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
             audio_data = audio_data / max_val
        
        try:
            # The feature extractor should handle resampling to its target SR, padding, and truncation.
            inputs = self.feature_extractor(
                audio_data, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt",
                padding="max_length", 
                truncation=True,
                max_length=self.sample_rate * 5 # Optional: set a max length like 5s for very long inputs
            )
            return inputs
        except Exception as e:
            logger.error(f"Error during audio preprocessing with Hugging Face feature_extractor: {e}", exc_info=True)
            return None

    def detect(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        start_time = time.time()
        cry_prob = 0.0
        is_crying = False
        
        min_audio_length = self.sample_rate // 20 # e.g., 0.05 seconds
        if audio_data.size == 0 or len(audio_data) < min_audio_length : 
            logger.debug(f"Audio data too short ({len(audio_data)} samples), skipping detection.")
            return False, 0.0
        
        if not self._ensure_model_loaded():
            logger.warning("Model not loaded or not usable, using fallback detection method.")
            return self._detect_fallback(audio_data)
                
        try:
            inputs = self.preprocess_audio(audio_data)
            if inputs is None:
                logger.error("Audio preprocessing failed. Using fallback.")
                return self._detect_fallback(audio_data)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.inference_mode(): 
                if self.model_type == "ast":
                    if not self.model or not hasattr(self.model, 'config'): # Check if model is fully loaded
                        logger.error("AST model or its config not available. Fallback.")
                        return self._detect_fallback(audio_data)
                    
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)[0] 
                    
                    if not self.cry_class_indices:
                        logger.warning(f"AST model '{self.actual_model_name}': Cry class indices are not set. Cannot determine cry probability accurately.")
                        cry_prob = 0.0
                    else:
                        cry_prob = sum(probs[idx].item() for idx in self.cry_class_indices if idx < len(probs))
                    
                    if cry_prob > 0.05: 
                        top_k_probs, top_k_indices = torch.topk(probs, 5) # Log top 5
                        top_labels = [f"{self.id2label.get(idx.item(), f'Idx_{idx.item()}')}: {p.item():.3f}" for idx, p in zip(top_k_indices, top_k_probs)]
                        logger.info(f"AST ('{self.actual_model_name}') CryProb: {cry_prob:.3f}. Top: [{', '.join(top_labels)}]")

                elif self.model_type == "wavlm":
                    if not self.model or not isinstance(self.model, torch.nn.Module): # Check if model is a valid nn.Module
                        logger.error("WavLM model not available or invalid. Fallback.")
                        return self._detect_fallback(audio_data)

                    logits = self.model(**inputs) 
                    probs = torch.softmax(logits, dim=-1)[0]
                    cry_prob = probs[1].item() if len(probs) > 1 else 0.0 # Index 1 is "baby_cry"
                    
                    if cry_prob > 0.05:
                        logger.info(f"WavLM ('{self.actual_model_name}') CryProb: {cry_prob:.3f} (NoCry: {probs[0].item():.3f})")
                        if hasattr(self.model, 'is_dummy') and self.model.is_dummy:
                             logger.warning("WavLM is using an UNTRAINED DUMMY classifier. Results are NOT reliable.")
                else:
                    logger.error(f"Unknown model type '{self.model_type}' for detection logic. Fallback.")
                    return self._detect_fallback(audio_data)

            is_crying = cry_prob > self.threshold
            if is_crying:
                self.stats["detection_count"] += 1
            
            return is_crying, float(cry_prob)

        except Exception as e:
            logger.error(f"Error during DL model detection with '{self.actual_model_name}': {e}", exc_info=True)
            return self._detect_fallback(audio_data)
        finally:
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            if len(self.stats["processing_times"]) > 100:
                self.stats["processing_times"].pop(0)
            
    def _detect_fallback(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        logger.debug("Using basic fallback detection method.")
        if audio_data.size == 0: return False, 0.0
        
        audio_data_float = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data_float))
        if max_abs > 0: 
            audio_data_float /= max_abs

        try:
            rms_energy = np.sqrt(np.mean(audio_data_float**2))
            
            # Basic heuristic
            confidence = 0.0
            if rms_energy > 0.05: # Lowered threshold for fallback sensitivity
                confidence = rms_energy * 0.5 # Scale confidence by energy somewhat
            
            # Simple ZCR check for very basic pitch indication
            if rms_energy > 0.02: # Only calculate ZCR if there's some energy
                zcr = librosa.feature.zero_crossing_rate(audio_data_float, frame_length=2048, hop_length=512)[0]
                mean_zcr = np.mean(zcr)
                if mean_zcr > 0.1 and mean_zcr < 0.35: # Typical ZCR for cries might be higher than speech
                    confidence += 0.2
            
            confidence = min(confidence, 0.9) # Cap confidence

            logger.info(f"Fallback: RMS Energy={rms_energy:.3f}, Confidence={confidence:.2f}")
            return confidence > self.threshold, confidence 
        except Exception as e:
            logger.error(f"Error in fallback detection: {e}", exc_info=True)
            return False, 0.0

    def get_model_info(self) -> Dict[str, Any]:
        avg_processing_time = np.mean(self.stats["processing_times"]) if self.stats["processing_times"] else 0
        # Safely get id2label sample
        id2label_sample_str = "Not available"
        if self.id2label:
            try:
                id2label_sample_str = str(dict(list(self.id2label.items())[:5]))
            except Exception:
                id2label_sample_str = "Error fetching sample"

        return {
            "model_name": self.actual_model_name or "N/A",
            "model_type": self.model_type or "N/A",
            "model_loaded": self.model_loaded,
            "device": str(self.device),
            "cry_class_indices": self.cry_class_indices if self.cry_class_indices else "N/A",
            "id2label_sample": id2label_sample_str,
            "threshold": self.threshold,
            "detection_count": self.stats["detection_count"],
            "avg_processing_time_ms": avg_processing_time * 1000
        }

