"""
Core detector class that integrates all components for baby cry detection.
"""
import numpy as np
import time
import threading
import sounddevice as sd
import os
import matplotlib.pyplot as plt
from queue import Queue
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Union

from detection.base import BaseDetector
from detection.simple import SimpleLevelDetector
from detection.advanced import AdvancedFeatureDetector
from detection.ml import MLDetector
from audio.player import AudioPlayer
from audio.processing import extract_audio_features, preprocess_audio

class BabyCryDetector:
    """Baby cry detector that uses different detection methods and plays audio on detection."""
    
    def __init__(self, voice_files: Optional[List[str]] = None, 
                detection_method: str = "simple",
                model_path: Optional[str] = None,
                threshold: float = 0.7, 
                simple_threshold: float = 0.4,
                playback_mode: str = "random"):
        """
        Initialize the baby cry detector with support for multiple detection methods.
        
        Args:
            voice_files: List of voice file paths or a single file path
            detection_method: Detection method to use ("simple", "advanced", "ml")
            model_path: Path to ML model file (required for "ml" method)
            threshold: Detection threshold for advanced and ML detection
            simple_threshold: Threshold for simple detection
            playback_mode: How to select files for playback ("random", "sequential", "all")
        """
        # Basic properties
        self.is_running = False
        self.audio_queue = Queue()
        self.last_detected = 0
        self.cooldown = 5  # Seconds between detections
        self.status_message = "Waiting to start..."
        self.detection_count = 0
        self.detection_history = []
        self.current_audio_level = 0
        self.is_playing_audio = False
        self.mic_error = None
        self.current_mic_name = "Unknown"
        self.available_mics = []
        self.selected_mic_id = None
        self.sample_rate = 22050
        self.audio_data_for_plot = np.zeros(1000)
        
        # Set thresholds
        self.threshold = threshold
        self.simple_threshold = simple_threshold
        
        # Audio file handling
        self.voice_files = []
        if voice_files:
            if isinstance(voice_files, str):  # Single file
                self.voice_files = [voice_files]
            else:  # Multiple files
                self.voice_files = voice_files
        else:
            # Default file
            self.voice_files = ["voce_madre.mp3"]
            
        # Current file index for sequential playback
        self.current_file_index = 0
        
        # Playback mode
        self.playback_mode = playback_mode
        
        # Initialize audio player
        self.audio_player = AudioPlayer()
        
        # Setup detector based on method
        self.detection_method = detection_method
        self._setup_detector(detection_method, model_path)
        
        # Initialize audio and mic
        self._scan_microphones()
    
    def _setup_detector(self, method: str, model_path: Optional[str] = None) -> None:
        """
        Setup the appropriate detector based on method.
        
        Args:
            method: Detection method ("simple", "advanced", "ml")
            model_path: Path to ML model file (required for "ml" method)
        """
        if method == "simple":
            self.detector = SimpleLevelDetector(threshold=self.simple_threshold)
        elif method == "advanced":
            self.detector = AdvancedFeatureDetector(threshold=self.threshold)
        elif method == "ml":
            if not model_path:
                raise ValueError("Model path is required for ML detection")
            # MLDetector will create a dummy model if the file doesn't exist
            self.detector = MLDetector(model_path=model_path, threshold=self.threshold, 
                                      create_dummy_model=True)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def get_detector_name(self) -> str:
        """Get the current detector's name in a user-friendly format."""
        detector_class = self.detector.__class__.__name__
        if detector_class == "SimpleLevelDetector":
            return "Simple level detection"
        elif detector_class == "AdvancedFeatureDetector":
            return "Advanced feature detection"
        elif detector_class == "MLDetector":
            return "ML-based detection"
        else:
            return detector_class
    
    def _scan_microphones(self) -> List[Tuple[int, str]]:
        """
        Scan for available microphones and get details.
        
        Returns:
            list: List of (microphone_id, microphone_name) tuples
        """
        try:
            devices = sd.query_devices()
            self.available_mics = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Input device
                    self.available_mics.append((i, device['name']))
            
            # Get default input device
            default_input = sd.query_devices(kind='input')
            self.current_mic_name = default_input['name']
            self.selected_mic_id = default_input['index'] if 'index' in default_input else None
            
            # Test microphone
            self._test_microphone()
            
            return self.available_mics
        except Exception as e:
            self.mic_error = str(e)
            print(f"Microphone scanning failed: {e}")
            return []
            
    def _test_microphone(self) -> None:
        """Test microphone access and functionality."""
        try:
            temp_stream = sd.InputStream(channels=1, samplerate=self.sample_rate, 
                                        device=self.selected_mic_id)
            temp_stream.start()
            time.sleep(0.1)
            temp_stream.stop()
            temp_stream.close()
            self.mic_error = None
        except Exception as e:
            self.mic_error = str(e)
            print(f"Microphone test failed: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """
        Callback for audio recording.
        
        Args:
            indata: Audio input data
            frames: Number of frames
            time_info: Time information
            status: Status information
        """
        if status:
            self.status_message = f"Audio error: {status}"
        self.audio_queue.put(indata[:, 0].copy())
    
    def _analyzer_thread(self) -> None:
        """Analyze audio in a separate thread."""
        while self.is_running:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                
                # Update audio data for visualization
                self._update_audio_visualization(audio_data)
                
                # Calculate audio level for visualization
                self.current_audio_level = min(np.sqrt(np.mean(audio_data**2)) * 100, 1.0)
                
                # Preprocess audio
                processed_audio = preprocess_audio(audio_data)
                
                # Detect crying using the selected detector
                is_crying, confidence = self.detector.detect(processed_audio)
                
                # Update status message
                self._update_status_message(confidence)
                
                if is_crying:
                    current_time = time.time()
                    if current_time - self.last_detected > self.cooldown:
                        self._handle_cry_detection(confidence)
            
            time.sleep(0.1)
    
    def _update_audio_visualization(self, audio_data: np.ndarray) -> None:
        """
        Update audio data for visualization.
        
        Args:
            audio_data: New audio data chunk
        """
        self.audio_data_for_plot = np.roll(self.audio_data_for_plot, -len(audio_data))
        if len(audio_data) < len(self.audio_data_for_plot):
            self.audio_data_for_plot[-len(audio_data):] = audio_data[:len(audio_data)]
        else:
            self.audio_data_for_plot = audio_data[-len(self.audio_data_for_plot):]
    
    def _update_status_message(self, confidence: float) -> None:
        """
        Update status message based on detection method.
        
        Args:
            confidence: Detection confidence level
        """
        detector_class = self.detector.__class__.__name__
        
        if detector_class == "SimpleLevelDetector":
            self.status_message = f"Audio level: {self.current_audio_level:.2f} (threshold: {self.simple_threshold:.2f})"
        elif detector_class == "MLDetector":
            self.status_message = f"ML prediction confidence: {confidence:.2f} (threshold: {self.threshold:.2f})"
        else:
            self.status_message = f"Detection confidence: {confidence:.2f} (threshold: {self.threshold:.2f})"
    
    def _handle_cry_detection(self, confidence: float) -> None:
        """
        Handle a positive cry detection.
        
        Args:
            confidence: Detection confidence level
        """
        detection_time = datetime.now()
        self.status_message = "🚨 Crying detected! Playing soothing audio..."
        self.detection_count += 1
        
        # Store detection info
        self.detection_history.append((detection_time, confidence))
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)
            
        # Play audio in separate thread
        self._play_audio()
        self.last_detected = time.time()
    
    def _play_audio(self) -> None:
        """Play audio when crying is detected."""
        # Create stop event for the player
        stop_event = threading.Event()
        
        def on_start(filename):
            self.is_playing_audio = True
            self.status_message = f"🔊 Playing: {filename}"
        
        def on_complete(message):
            self.is_playing_audio = False
            self.status_message = "Listening for baby crying..."
        
        # Start playback in a separate thread
        play_thread = threading.Thread(
            target=self.audio_player.play_files,
            args=(self.voice_files, self.playback_mode, self.current_file_index),
            kwargs={
                'on_start': on_start,
                'on_complete': on_complete,
                'stop_event': stop_event
            }
        )
        play_thread.daemon = True
        play_thread.start()
        
        # Update file index for sequential mode
        if self.playback_mode == "sequential":
            self.current_file_index = (self.current_file_index + 1) % len(self.voice_files)
    
    def create_audio_plot(self) -> plt.Figure:
        """
        Create an audio waveform plot with real-time level indicator.
        
        Returns:
            matplotlib.Figure: Plot figure
        """
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        
        # Plot audio waveform
        x = np.arange(len(self.audio_data_for_plot))
        ax.plot(x, self.audio_data_for_plot, color='blue', alpha=0.7)
        
        # Add threshold indicator based on detection method
        detector_class = self.detector.__class__.__name__
        if detector_class == "SimpleLevelDetector":
            threshold_line = self.simple_threshold / 50  # Scale to fit in plot range [-1, 1]
            ax.axhline(y=threshold_line, color='red', linestyle='--', alpha=0.7, 
                      label=f"Threshold ({self.simple_threshold:.2f})")
            ax.legend(loc='upper right', fontsize='x-small')
        
        # Add current audio level indicator with color coding
        level = self.current_audio_level
        if level < 0.3:
            level_color = 'green'
        elif level < 0.7:
            level_color = 'orange'
        else:
            level_color = 'red'
            
        # Add level bar at the bottom
        level_width = len(self.audio_data_for_plot) * (level)
        rect = plt.Rectangle((0, -0.95), level_width, 0.1, color=level_color, alpha=0.7)
        ax.add_patch(rect)
        
        # Add level text
        ax.text(len(self.audio_data_for_plot)*0.05, -0.9, 
                f"Level: {level:.2f}", color='black', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Format plot
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, len(self.audio_data_for_plot))
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        
        # Add detection method and microphone info
        detector_name = self.get_detector_name()
        ax.text(0.02, 0.95, f"Mode: {detector_name} | Mic: {self.current_mic_name}", 
                transform=ax.transAxes, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Add audio files count
        ax.text(0.02, 0.90, f"Audio files: {len(self.voice_files)} ({self.playback_mode} mode)", 
                transform=ax.transAxes, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Highlight when crying is detected
        if time.time() - self.last_detected < 3:
            ax.set_facecolor((1.0, 0.9, 0.9))
            ax.text(0.5, 0.5, 'Crying detected!', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes,
                    color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def set_microphone(self, mic_id) -> str:
        """
        Set the microphone to use.
        
        Args:
            mic_id: Microphone ID or "Default Microphone"
            
        Returns:
            str: Status message
        """
        if self.is_running:
            self.stop_detection()
            time.sleep(0.5)
        
        try:
            if mic_id == "Default Microphone" or mic_id is None:
                self.selected_mic_id = None
                default_input = sd.query_devices(kind='input')
                self.current_mic_name = default_input['name']
            else:
                # Find the mic in available_mics
                for id, name in self.available_mics:
                    if str(id) == str(mic_id):
                        self.selected_mic_id = id
                        self.current_mic_name = name
                        break
            
            self._test_microphone()
            return f"Microphone set to: {self.current_mic_name}"
        except Exception as e:
            self.mic_error = str(e)
            return f"Error setting microphone: {e}"

    def start_detection(self, voice_files=None, detection_method=None, model_path=None,
                        threshold=None, simple_threshold=None, playback_mode=None, 
                        mic_id=None) -> Dict[str, Any]:
        """
        Start the detector with updated parameters.
        
        Args:
            voice_files: List of voice files or single file path
            detection_method: Detection method ("simple", "advanced", "ml")
            model_path: Path to ML model file
            threshold: Detection threshold for advanced/ML methods
            simple_threshold: Detection threshold for simple method
            playback_mode: Playback mode for audio files
            mic_id: Microphone ID
            
        Returns:
            dict: Status values
        """
        if self.is_running:
            self.stop_detection()
            time.sleep(0.5)
        
        # Set microphone if specified
        if mic_id is not None and mic_id != "":
            self.set_microphone(mic_id)
        
        # Update voice files if provided
        if voice_files is not None:
            if isinstance(voice_files, list):
                self.voice_files = voice_files
            else:
                self.voice_files = [voice_files]
            
            # Reset current file index
            self.current_file_index = 0
        
        # Update thresholds if provided
        if threshold is not None:
            self.threshold = float(threshold)
        if simple_threshold is not None:
            self.simple_threshold = float(simple_threshold)
            
        # Update playback mode if provided
        if playback_mode is not None:
            self.playback_mode = playback_mode
        
        # Update detector if method is provided
        if detection_method is not None:
            self.detection_method = detection_method
            self._setup_detector(detection_method, model_path)
        
        # Reset detection state
        self.is_running = True
        self.detection_count = 0
        self.detection_history = []
        
        # Check if files exist
        missing_files = [f for f in self.voice_files if not os.path.exists(f)]
        if missing_files:
            self.status_message = f"Warning: {len(missing_files)} audio file(s) not found"
        
        # Start analyzer thread
        analyzer = threading.Thread(target=self._analyzer_thread)
        analyzer.daemon = True
        analyzer.start()
        
        # Try to start audio stream
        try:
            self.stream = sd.InputStream(callback=self._audio_callback, 
                                        channels=1, 
                                        samplerate=self.sample_rate,
                                        device=self.selected_mic_id)
            self.stream.start()
            
            # Update status message with detection method
            detector_name = self.get_detector_name()
            self.status_message = f"Baby cry detector active using {detector_name}. Listening via {self.current_mic_name}..."
        except Exception as e:
            self.mic_error = str(e)
            self.status_message = f"Recording error: {e}"
            self.is_running = False
        
        return self.get_status_values()

    def stop_detection(self) -> Dict[str, Any]:
        """
        Stop the detector.
        
        Returns:
            dict: Status values
        """
        if not self.is_running:
            self.status_message = "Detector is not active"
            return self.get_status_values()
            
        self.is_running = False
        
        # Stop audio playback if active
        if self.is_playing_audio:
            self.audio_player.stop()
            
        # Stop audio stream
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.status_message = f"Error stopping stream: {e}"
                
        self.status_message = "Detector stopped"
        return self.get_status_values()

    def get_status_values(self) -> Dict[str, Any]:
        """
        Return current status information.
        
        Returns:
            dict: Status values dictionary
        """
        running_status = "🟢 ACTIVE" if self.is_running else "🔴 INACTIVE"
        detection_count = f"Crying detected: {self.detection_count}"
        
        # Audio level display with visual indicator
        level = int(self.current_audio_level * 100)
        # Add color coding based on level
        if level < 30:
            level_color = "🟢"  # Green for low
        elif level < 70:
            level_color = "🟡"  # Yellow for medium
        else:
            level_color = "🔴"  # Red for high
            
        progress_bar = f"Audio level: {level_color} {'█' * (level // 5)}{'░' * (20 - level // 5)} {level}%"
        
        # Detection method info
        detection_method = self.get_detector_name()
        
        # Get appropriate threshold based on detector type
        detector_class = self.detector.__class__.__name__
        if detector_class == "SimpleLevelDetector":
            threshold_info = f"Threshold: {self.simple_threshold:.2f}"
        else:
            threshold_info = f"Threshold: {self.threshold:.2f}"
        
        # Add playback info
        playing_status = ""
        if self.is_playing_audio:
            playing_status = f"🔊 PLAYING AUDIO ({self.playback_mode.upper()} MODE)"
        
        # Detection history
        history_text = "Recent detections:"
        if not self.detection_history:
            history_text += " None"
        else:
            for dt, conf in self.detection_history[-5:]:
                timestamp = dt.strftime("%H:%M:%S")
                metric = "level" if detector_class == "SimpleLevelDetector" else "confidence"
                history_text += f"\n - {timestamp} ({metric}: {conf:.2f})"
        
        # Files info
        file_text = f"Audio files ({len(self.voice_files)}):"
        if not self.voice_files:
            file_text += " None"
        else:
            for i, file in enumerate(self.voice_files):
                filename = os.path.basename(file)
                if i < 5:  # Show only first 5 files
                    next_indicator = " (next)" if i == self.current_file_index and self.playback_mode == "sequential" else ""
                    file_text += f"\n - {filename}{next_indicator}"
            if len(self.voice_files) > 5:
                file_text += f"\n - ... and {len(self.voice_files) - 5} more"
        
        # Create visualization
        audio_viz = self.create_audio_plot()
        
        # Microphone status
        mic_status = f"⚠️ Microphone error: {self.mic_error}" if self.mic_error else f"✅ Microphone: {self.current_mic_name}"
        
        return {
            "running_status": running_status,
            "status_message": self.status_message,
            "detection_count": detection_count,
            "progress_bar": progress_bar,
            "playing_status": playing_status,
            "history_text": history_text,
            "mic_status": mic_status,
            "audio_viz": audio_viz,
            "detection_method": detection_method,
            "threshold_info": threshold_info,
            "file_text": file_text
        }
    
    def get_available_mics(self) -> List[Tuple[str, str]]:
        """
        Get list of available microphones for the dropdown.
        
        Returns:
            list: List of (id, name) tuples for microphones
        """
        self._scan_microphones()
        choices = [("Default Microphone", "Default Microphone")]
        for id, name in self.available_mics:
            choices.append((str(id), f"{name}"))
        return choices