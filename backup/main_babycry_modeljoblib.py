import numpy as np
import librosa
import time
import pygame
import gradio as gr
from gradio import update as gr_update
import threading
import sounddevice as sd
import os
import matplotlib.pyplot as plt
from queue import Queue
from datetime import datetime
import sys

# Ensure matplotlib doesn't open interactive windows
plt.ioff()

class BabyCryDetector:
    def __init__(self, mother_voice_file="voce_madre.mp3", threshold=0.7, use_simple_detection=False, 
                 simple_threshold=0.4):
        """Initialize the baby cry detector."""
        self.threshold = threshold
        self.mother_voice_file = mother_voice_file
        self.is_running = False
        self.audio_queue = Queue()
        self.last_detected = 0
        self.cooldown = 5
        self.status_message = "Waiting to start..."
        self.detection_count = 0
        self.detection_history = []
        self.current_audio_level = 0
        self.is_playing_audio = False
        self.mic_error = None
        self.current_mic_name = "Unknown"
        self.available_mics = []
        self.selected_mic_id = None
        
        # Add simple detection option
        self.use_simple_detection = use_simple_detection
        self.simple_threshold = simple_threshold
        
        self.sample_rate = 22050
        self.duration = 2
        self.buffer_size = self.sample_rate * 2
        self.audio_data_for_plot = np.zeros(1000)
        import joblib
        self.model = joblib.load("model.joblib")
        
        self._init_audio_player()
        self._scan_microphones()

    def _scan_microphones(self):
        """Scan for available microphones and get details."""
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

    def _test_microphone(self):
        """Test microphone access."""
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

    def _init_audio_player(self):
        """Initialize pygame mixer."""
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100)
            return True
        except Exception as e:
            self.status_message = f"Audio player initialization error: {e}"
            return False

    def _extract_features(self, audio):
        """Extract audio features for cry detection."""
        if len(audio) == 0:
            return {"rms_mean": 0, "zero_crossing_rate_mean": 0, "zero_crossing_rate_std": 0, "centroid_mean": 0}
        
        rms = librosa.feature.rms(y=audio)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        return {
            "rms_mean": np.mean(rms),
            "zero_crossing_rate_mean": np.mean(zcr),
            "zero_crossing_rate_std": np.std(zcr),
            "centroid_mean": np.mean(spec_cent) if len(spec_cent) > 0 else 0
        }

    def _is_baby_crying(self, audio):
        if len(audio) == 0:
            return False, 0.0

        self.current_audio_level = min(np.sqrt(np.mean(audio**2)) * 100, 1.0)

        self.audio_data_for_plot = np.roll(self.audio_data_for_plot, -len(audio))
        if len(audio) < len(self.audio_data_for_plot):
            self.audio_data_for_plot[-len(audio):] = audio[:len(audio)]
        else:
            self.audio_data_for_plot = audio[-len(self.audio_data_for_plot):]

        if self.use_simple_detection:
            is_crying = self.current_audio_level > self.simple_threshold
            return is_crying, self.current_audio_level

        features = self._extract_features(audio)
        feature_vector = np.array([
            features["rms_mean"],
            features["zero_crossing_rate_mean"],
            features["zero_crossing_rate_std"],
            features["centroid_mean"]
        ]).reshape(1, -1)

        try:
            confidence = self.model.predict_proba(feature_vector)[0][1]
        except Exception as e:
            print(f"Prediction error: {e}")
            confidence = 0.0

        return confidence > self.threshold, confidence


    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio recording."""
        if status:
            self.status_message = f"Audio error: {status}"
        self.audio_queue.put(indata[:, 0].copy())

    def _analyzer_thread(self):
        """Analyze audio in a separate thread."""
        while self.is_running:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                is_crying, confidence = self._is_baby_crying(audio_data)
                
                # Update status message with different text based on detection method
                if self.use_simple_detection:
                    self.status_message = f"Audio level: {self.current_audio_level:.2f} (threshold: {self.simple_threshold:.2f})"
                else:
                    self.status_message = f"Detection confidence: {confidence:.2f} (threshold: {self.threshold:.2f})"
                
                if is_crying:
                    current_time = time.time()
                    if current_time - self.last_detected > self.cooldown:
                        detection_time = datetime.now()
                        self.status_message = "🚨 Crying detected! Playing mother's voice..."
                        self.detection_count += 1
                        # Store different confidence values based on detection method
                        conf_value = self.current_audio_level if self.use_simple_detection else confidence
                        self.detection_history.append((detection_time, conf_value))
                        if len(self.detection_history) > 10:
                            self.detection_history.pop(0)
                        play_thread = threading.Thread(target=self._play_mother_voice)
                        play_thread.daemon = True
                        play_thread.start()
                        self.last_detected = current_time
            time.sleep(0.1)

    def _play_mother_voice(self):
        """Play the mother's voice audio."""
        if not os.path.exists(self.mother_voice_file):
            self.status_message = f"Error: File '{self.mother_voice_file}' not found"
            return
        try:
            if not self._init_audio_player():
                return
            self.is_playing_audio = True
            pygame.mixer.music.load(self.mother_voice_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            self.is_playing_audio = False
        except Exception as e:
            self.status_message = f"Audio playback error: {e}"
            self.is_playing_audio = False

    def create_audio_plot(self):
        """Create an audio waveform plot with real-time level indicator."""
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        
        # Plot audio waveform
        x = np.arange(len(self.audio_data_for_plot))
        ax.plot(x, self.audio_data_for_plot, color='blue', alpha=0.7)
        
        # Add horizontal line for threshold if using simple detection
        if self.use_simple_detection:
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
        
        # Add microphone info
        ax.text(0.02, 0.95, f"Mic: {self.current_mic_name}", transform=ax.transAxes,
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        
        # Highlight when crying is detected
        if time.time() - self.last_detected < 3:
            ax.set_facecolor((1.0, 0.9, 0.9))
            ax.text(0.5, 0.5, 'Crying detected!', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes,
                    color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def set_microphone(self, mic_id):
        """Set the microphone to use."""
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

    def start_detection(self, mother_voice_file, threshold, use_simple_detection, simple_threshold, mic_id=None):
        """Start the detector with updated parameters."""
        if self.is_running:
            self.stop_detection()
            time.sleep(0.5)
        
        # Set microphone if specified
        if mic_id is not None and mic_id != "":
            self.set_microphone(mic_id)
        
        self.mother_voice_file = mother_voice_file
        self.threshold = threshold
        self.use_simple_detection = use_simple_detection
        self.simple_threshold = simple_threshold
        self.is_running = True
        self.detection_count = 0
        self.detection_history = []
        
        if not os.path.exists(self.mother_voice_file):
            self.status_message = f"Warning: Mother's voice file '{self.mother_voice_file}' not found"
        
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
            detection_mode = "simple level detection" if self.use_simple_detection else "advanced feature detection"
            self.status_message = f"Baby cry detector active using {detection_mode}. Listening via {self.current_mic_name}..."
        except Exception as e:
            self.mic_error = str(e)
            self.status_message = f"Recording error: {e}"
            self.is_running = False
        
        return self.get_status_values()

    def stop_detection(self):
        """Stop the detector."""
        if not self.is_running:
            self.status_message = "Detector is not active"
            return self.get_status_values()
        self.is_running = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.status_message = f"Error stopping stream: {e}"
        self.status_message = "Detector stopped"
        return self.get_status_values()

    def get_status_values(self):
        """Return current status for Gradio."""
        running_status = "🟢 ACTIVE" if self.is_running else "🔴 INACTIVE"
        detection_count = f"Crying detected: {self.detection_count}"
        
        # Improved audio level display
        level = int(self.current_audio_level * 100)
        # Add color coding based on level
        if level < 30:
            level_color = "🟢"  # Green for low
        elif level < 70:
            level_color = "🟡"  # Yellow for medium
        else:
            level_color = "🔴"  # Red for high
            
        progress_bar = f"Audio level: {level_color} {'█' * (level // 5)}{'░' * (20 - level // 5)} {level}%"
        
        # Add detection method info
        detection_method = "Simple level detection" if self.use_simple_detection else "Model-based detection (model.joblib)"
        threshold_info = f"Threshold: {self.simple_threshold:.2f}" if self.use_simple_detection else f"Threshold: {self.threshold:.2f}"
        
        playing_status = "🔊 PLAYING MOTHER'S VOICE" if self.is_playing_audio else ""
        
        # Enhanced history with detection method
        history_text = "Recent detections:"
        if not self.detection_history:
            history_text += " None"
        else:
            for dt, conf in self.detection_history[-5:]:
                timestamp = dt.strftime("%H:%M:%S")
                metric = "level" if self.use_simple_detection else "confidence"
                history_text += f"\n - {timestamp} ({metric}: {conf:.2f})"
        
        audio_viz = self.create_audio_plot()
        
        # Enhanced microphone status with name
        mic_status = f"⚠️ Microphone error: {self.mic_error}" if self.mic_error else f"✅ Microphone: {self.current_mic_name}"
        
        return (running_status, self.status_message, detection_count, progress_bar,
                playing_status, history_text, mic_status, audio_viz, 
                detection_method, threshold_info)

    def refresh_status(self):
        """Refresh status manually."""
        return self.get_status_values()

    def get_available_mics(self):
        """Get list of available microphones for the dropdown."""
        self._scan_microphones()
        choices = [("Default Microphone", "Default Microphone")]
        for id, name in self.available_mics:
            choices.append((str(id), f"{name}"))
        return choices

# Initialize detector
detector = BabyCryDetector()

# Gradio interface functions
def start_detector(file_path, threshold, use_simple_detection, simple_threshold, selected_mic):
    return detector.start_detection(file_path, float(threshold), use_simple_detection, float(simple_threshold), selected_mic)

def stop_detector():
    return detector.stop_detection()

def refresh_status():
    return detector.refresh_status()

def get_microphones():
    mic_choices = detector.get_available_mics()
    return gr_update(choices=mic_choices)

def toggle_detection_method(use_simple):
    return not use_simple

# Build Gradio interface
with gr.Blocks(title="Baby Cry Detector") as app:
    gr.Markdown("# 👶 Baby Cry Detector")
    gr.Markdown("Listens through the microphone and plays the mother's voice when a baby cry is detected.")
    
    with gr.Row():
        with gr.Column(scale=2):
            voice_file = gr.Audio(label="Upload mother's voice recording", type="filepath", value="voce_madre.mp3")
            
            # Detection methods
            with gr.Group():
                gr.Markdown("### Detection Method")
                use_simple = gr.Checkbox(value=True, label="Use simple level detection (≥ 40%)", 
                                       info="Triggers when audio level exceeds threshold")
                simple_threshold_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.05,
                                                  label="Simple detection level threshold",
                                                  info="Triggered when audio level exceeds this value (0.4 = 40%)")
                
                use_advanced = gr.Checkbox(value=False, label="Use advanced feature detection", 
                                         info="Uses audio features like pitch and modulation to detect crying")
                threshold_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05,
                                          label="Advanced detection sensitivity",
                                          info="Lower: more sensitive. Higher: less sensitive.")
            
            # Microphone selection
            with gr.Group():
                gr.Markdown("### Microphone")
                mic_dropdown = gr.Dropdown(label="Select Microphone", 
                                         choices=[("Default Microphone", "Default Microphone")],
                                         value="Default Microphone",
                                         info="Choose which microphone to use")
                refresh_mics_btn = gr.Button("🔍 Refresh Microphone List")
        
        with gr.Column(scale=2):
            gr.Markdown("### Real-time Status")
            running_status = gr.Textbox(label="", value="🔴 INACTIVE")
            status_message = gr.Textbox(label="", value="Waiting to start...")
            detection_counter = gr.Textbox(label="", value="Crying detected: 0")
            audio_level_indicator = gr.Textbox(label="", value="Audio level: ░░░░░░░░░░░░░░░░░░░░ 0%")
            playing_status = gr.Textbox(label="", value="")
            detection_method = gr.Textbox(label="Detection Method", value="Simple level detection")
            threshold_info = gr.Textbox(label="Threshold", value="Threshold: 0.4")
            mic_status = gr.Textbox(label="Microphone Status", value="Checking microphone...")
            
            # Visualization
            gr.Markdown("### Audio Waveform")
            audio_viz = gr.Plot(label="")
            
            # History
            detection_history = gr.Textbox(label="Detection History", value="Recent detections: None", lines=5)
    
    with gr.Row():
        start_btn = gr.Button("▶️ Start Detector", variant="primary", scale=2)
        stop_btn = gr.Button("⏹️ Stop Detector", variant="stop", scale=2)
        refresh_btn = gr.Button("🔄 Refresh Status", elem_id="refresh_btn", scale=1)
    
    gr.Markdown("""
    ### Instructions
    1. Upload a mother's voice recording or use the default
    2. Choose detection method:
       - **Simple Level Detection**: Triggers when audio level exceeds 40% (or set threshold)
       - **Advanced Feature Detection**: Uses audio features for more accurate baby cry detection
    3. Select your microphone from the dropdown
    4. Click "Start Detector" to begin listening
    5. The status and audio waveform update in real-time
    6. Click "Stop Detector" when finished
    """)
    
    # Logic for checkboxes to be mutually exclusive
    use_simple.change(fn=lambda x: not x, inputs=use_simple, outputs=use_advanced)
    use_advanced.change(fn=lambda x: not x, inputs=use_advanced, outputs=use_simple)
    
    # Button connections
    start_btn.click(fn=start_detector, 
                   inputs=[voice_file, threshold_slider, use_simple, simple_threshold_slider, mic_dropdown],
                   outputs=[running_status, status_message, detection_counter, audio_level_indicator,
                           playing_status, detection_history, mic_status, audio_viz,
                           detection_method, threshold_info])
    
    stop_btn.click(fn=stop_detector, 
                  inputs=[], 
                  outputs=[running_status, status_message, detection_counter, audio_level_indicator, 
                           playing_status, detection_history, mic_status, audio_viz,
                           detection_method, threshold_info])
    
    refresh_btn.click(fn=refresh_status, 
                     inputs=[], 
                     outputs=[running_status, status_message, detection_counter, audio_level_indicator, 
                              playing_status, detection_history, mic_status, audio_viz,
                              detection_method, threshold_info])
    
    refresh_mics_btn.click(fn=get_microphones, inputs=[], outputs=[mic_dropdown])
    
    # Auto-refresh every second using JavaScript
    gr.HTML("""
    <script>
    setInterval(function() {
        document.getElementById('refresh_btn').click();
    }, 1000);
    </script>
    """)

# Launch the app
if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Gradio version: {gr.__version__}")
    print(f"Numpy version: {np.__version__}")
    print(f"Pygame version: {pygame.version.ver}")
    print(f"Available microphones:")
    for id, name in detector.get_available_mics():
        print(f" - {name} (ID: {id})")
    app.launch(share=False)