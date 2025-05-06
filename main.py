"""
Baby Cry Detector - Main Application

A Gradio-based application for detecting baby crying and playing soothing audio in response.
Supports multiple detection methods including ML-based detection with a pre-trained model.
"""
import os
import time
import sys
import numpy as np
import gradio as gr
from gradio import update as gr_update
import threading
import matplotlib.pyplot as plt

# Ensure plotting works without GUI
plt.ioff()

# Import modules from project structure
from core.detector import BabyCryDetector
from utils.helpers import (
    process_uploaded_files,
    add_files_to_list,
    remove_file_from_list,
    clear_files,
    print_system_info
)

# Default paths
DEFAULT_MODEL_PATH = os.path.join("models", "baby_cry_svm.joblib")
DEFAULT_AUDIO_FILE = "voce_madre.mp3"

# Create detector instance
detector = BabyCryDetector()

# Gradio interface functions
def start_detector(files, detection_method, threshold, simple_threshold, playback_mode, selected_mic):
    """Start the detector with the selected files and settings."""
    file_list = process_uploaded_files(files)
    
    # Determine model path if ML method selected
    model_path = None
    if detection_method == "ml":
        model_path = DEFAULT_MODEL_PATH
        # Model will be created if it doesn't exist - see MLDetector implementation
    
    # Start the detector
    status = detector.start_detection(
        voice_files=file_list,
        detection_method=detection_method,
        model_path=model_path,
        threshold=float(threshold),
        simple_threshold=float(simple_threshold),
        playback_mode=playback_mode,
        mic_id=selected_mic
    )
    
    # Extract and return values in correct order for Gradio
    return (
        status["running_status"],
        status["status_message"],
        status["detection_count"],
        status["progress_bar"],
        status["playing_status"],
        status["history_text"],
        status["mic_status"],
        status["audio_viz"],
        status["detection_method"],
        status["threshold_info"],
        status["file_text"],
        gr_update(value=detector.voice_files)
    )

def stop_detector():
    """Stop the detector."""
    status = detector.stop_detection()
    
    return (
        status["running_status"],
        status["status_message"],
        status["detection_count"],
        status["progress_bar"],
        status["playing_status"],
        status["history_text"],
        status["mic_status"],
        status["audio_viz"],
        status["detection_method"],
        status["threshold_info"],
        status["file_text"],
        gr_update(value=detector.voice_files)
    )

def refresh_status():
    """Refresh the status display."""
    status = detector.get_status_values()
    
    return (
        status["running_status"],
        status["status_message"],
        status["detection_count"],
        status["progress_bar"],
        status["playing_status"],
        status["history_text"],
        status["mic_status"],
        status["audio_viz"],
        status["detection_method"],
        status["threshold_info"],
        status["file_text"],
        gr_update(value=detector.voice_files)
    )

def get_microphones():
    """Get available microphones."""
    mic_choices = detector.get_available_mics()
    return gr_update(choices=mic_choices)

def add_files_to_detector(files):
    """Add files to the detector."""
    result, updated_files = add_files_to_list(files, detector.voice_files)
    detector.voice_files = updated_files
    return result, gr_update(value=detector.voice_files)

def remove_file_from_detector(file_index):
    """Remove a file by index from the detector."""
    result, updated_files = remove_file_from_list(file_index, detector.voice_files)
    detector.voice_files = updated_files
    return result, gr_update(value=detector.voice_files)

def clear_all_files():
    """Clear all files from the detector."""
    result, updated_files = clear_files(detector.voice_files)
    detector.voice_files = updated_files
    return result, gr_update(value=[])

def toggle_detection_method(val):
    """Select mutually exclusive detection methods."""
    if val == "simple":
        return gr_update(value=False), gr_update(value=False)
    elif val == "advanced":
        return gr_update(value=False), gr_update(value=False)
    elif val == "ml":
        return gr_update(value=False), gr_update(value=False)
    return gr_update(value=False), gr_update(value=False)

# Build Gradio interface
with gr.Blocks(title="Baby Cry Detector") as app:
    gr.Markdown("# 👶 Baby Cry Detector")
    gr.Markdown("Listens through the microphone and plays soothing audio when a baby cry is detected.")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Audio file management
            with gr.Group():
                gr.Markdown("### Audio Files")
                voice_files = gr.File(label="Upload audio files", file_types=["audio"], file_count="multiple")
                
                with gr.Row():
                    add_btn = gr.Button("➕ Add Files")
                    clear_btn = gr.Button("🗑️ Clear All")
                
                file_list = gr.Dropdown(label="Loaded Audio Files", choices=[], allow_custom_value=True, multiselect=True)
                file_status = gr.Textbox(label="File Status", value="No files loaded")
                
                # Playback mode
                playback_mode = gr.Radio(
                    choices=["random", "sequential", "all"], 
                    value="random", 
                    label="Playback Mode",
                    info="How to play audio files when crying is detected"
                )
            
            # Detection methods
            with gr.Group():
                gr.Markdown("### Detection Method")
                detection_method = gr.Radio(
                    choices=["simple", "advanced", "ml"], 
                    value="simple", 
                    label="Detection Method",
                    info="Select which detection algorithm to use"
                )
                
                with gr.Row():
                    simple_threshold_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.4, 
                        step=0.05,
                        label="Simple detection level threshold",
                        info="Triggered when audio level exceeds this value (0.4 = 40%)"
                    )
                    
                    threshold_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.05,
                        label="Advanced/ML detection sensitivity",
                        info="Lower: more sensitive. Higher: less sensitive."
                    )
                
                ml_status = gr.Textbox(
                    label="ML Model Status", 
                    value=f"Model will be loaded from: {DEFAULT_MODEL_PATH}"
                )
            
            # Microphone selection
            with gr.Group():
                gr.Markdown("### Microphone")
                mic_dropdown = gr.Dropdown(
                    label="Select Microphone", 
                    choices=[("Default Microphone", "Default Microphone")],
                    value="Default Microphone",
                    info="Choose which microphone to use"
                )
                refresh_mics_btn = gr.Button("🔍 Refresh Microphone List")
        
        with gr.Column(scale=2):
            gr.Markdown("### Real-time Status")
            running_status = gr.Textbox(label="", value="🔴 INACTIVE")
            status_message = gr.Textbox(label="", value="Waiting to start...")
            detection_counter = gr.Textbox(label="", value="Crying detected: 0")
            audio_level_indicator = gr.Textbox(label="", value="Audio level: ░░░░░░░░░░░░░░░░░░░░ 0%")
            playing_status = gr.Textbox(label="", value="")
            detection_method_display = gr.Textbox(label="Detection Method", value="Simple level detection")
            threshold_info = gr.Textbox(label="Threshold", value="Threshold: 0.4")
            mic_status = gr.Textbox(label="Microphone Status", value="Checking microphone...")
            
            # Visualization
            gr.Markdown("### Audio Waveform")
            audio_viz = gr.Plot(label="")
            
            # History and files info
            with gr.Row():
                with gr.Column():
                    detection_history = gr.Textbox(label="Detection History", value="Recent detections: None", lines=5)
                with gr.Column():
                    file_info = gr.Textbox(label="Audio Files", value="Audio files (0): None", lines=5)
    
    with gr.Row():
        start_btn = gr.Button("▶️ Start Detector", variant="primary", scale=2)
        stop_btn = gr.Button("⏹️ Stop Detector", variant="stop", scale=2)
        refresh_btn = gr.Button("🔄 Refresh Status", elem_id="refresh_btn", scale=1)
    
    gr.Markdown("""
    ### Instructions
    1. Upload one or more audio files (lullabies, mother's voice, white noise, etc.)
    2. Choose a detection method:
       - **Simple Level Detection**: Triggered when audio level exceeds threshold
       - **Advanced Feature Detection**: Uses audio features for more accurate baby cry detection
       - **ML Detection**: Uses a pre-trained machine learning model for highest accuracy
    3. Choose a playback mode:
       - **Random**: Plays a random file each time
       - **Sequential**: Cycles through files in order
       - **All**: Plays all files one after another
    4. Select your microphone from the dropdown
    5. Click "Start Detector" to begin listening
    6. The status and audio waveform update in real-time
    7. Click "Stop Detector" when finished
    """)
    
    # Detection method change handler
    detection_method.change(
        fn=lambda x: (
            gr_update(visible=x=="simple"),
            gr_update(visible=x!="simple")
        ),
        inputs=[detection_method],
        outputs=[simple_threshold_slider, threshold_slider]
    )
    
    # Check ML model existence 
    def check_ml_model():
        if os.path.exists(DEFAULT_MODEL_PATH):
            return f"✅ ML model found: {DEFAULT_MODEL_PATH}"
        return f"ℹ️ ML model not found at: {DEFAULT_MODEL_PATH}. A dummy model will be created automatically."
    
    detection_method.change(
        fn=lambda x: check_ml_model() if x == "ml" else "",
        inputs=[detection_method],
        outputs=[ml_status]
    )
    
    # Button connections for file management
    add_btn.click(fn=add_files_to_detector, inputs=[voice_files], outputs=[file_status, file_list])
    clear_btn.click(fn=clear_all_files, inputs=[], outputs=[file_status, file_list])
    
    # Button connections for detector
    start_btn.click(
        fn=start_detector, 
        inputs=[
            file_list, 
            detection_method,
            threshold_slider,
            simple_threshold_slider, 
            playback_mode, 
            mic_dropdown
        ],
        outputs=[
            running_status, 
            status_message, 
            detection_counter, 
            audio_level_indicator,
            playing_status, 
            detection_history, 
            mic_status, 
            audio_viz,
            detection_method_display, 
            threshold_info, 
            file_info,
            file_list
        ]
    )
    
    stop_btn.click(
        fn=stop_detector, 
        inputs=[], 
        outputs=[
            running_status, 
            status_message, 
            detection_counter, 
            audio_level_indicator, 
            playing_status, 
            detection_history, 
            mic_status, 
            audio_viz,
            detection_method_display, 
            threshold_info, 
            file_info,
            file_list
        ]
    )
    
    refresh_btn.click(
        fn=refresh_status, 
        inputs=[], 
        outputs=[
            running_status, 
            status_message, 
            detection_counter, 
            audio_level_indicator, 
            playing_status, 
            detection_history, 
            mic_status, 
            audio_viz,
            detection_method_display, 
            threshold_info, 
            file_info,
            file_list
        ]
    )
    
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
    print_system_info()
    
    # Check if model directory exists and create if not
    os.makedirs("models", exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"Note: ML model not found at {DEFAULT_MODEL_PATH}")
        print("A dummy model will be created automatically when selecting ML detection.")
        print("For better accuracy, consider training a real model with train_model.py")
    
    app.launch(share=False)