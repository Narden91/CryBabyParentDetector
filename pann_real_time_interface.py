import gradio as gr
import threading
import queue
import numpy as np
import pyaudio
import random
import time
from panns_inference import SoundEventDetection, labels 
from pydub import AudioSegment
from pydub.playback import play
import os 
import traceback 


# ── Parametri Gradio ───────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch is not installed. PANNs inference will likely fail or run on CPU by default if model manages to load.")
    TORCH_AVAILABLE = False


# ── Parametri audio ─────────────────────────────────────────────
SR = 32000
WINDOW_SEC = 2.0
HOP_SEC = 1.0 # Using 1.0s hop for 1s overlap with 2s window
THRESHOLD = 0.2
WINDOW_SIZE = int(WINDOW_SEC * SR)
HOP_SIZE = int(HOP_SEC * SR)
try:
    BABY_IDX = labels.index("Baby cry, infant cry")
except ValueError:
    print("ERROR: 'Baby cry, infant cry' not found in labels. PANNs labels might be different.")
    print(f"Available labels (first 10): {labels[:10]}")
    BABY_IDX = -1 

# ── Modello (Load once) ─────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights/Cnn14_DecisionLevelMax_mAP=0.385.pth")
sed_model = None

if BABY_IDX != -1: # Only load model if Baby_IDX is valid
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: Weights file not found at {WEIGHTS_PATH}")
    else:
        try:
            device = "cpu"
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For Apple Silicon
                    device = "mps"
            
            sed_model = SoundEventDetection(
                checkpoint_path=WEIGHTS_PATH,
                device=device
            )
            print(f"PANNs model loaded successfully on {device}.")
        except Exception as e:
            print(f"Error loading PANNs model: {e}")
            traceback.print_exc()
else:
    print("Model not loaded due to BABY_IDX issue.")


# ── Funzione principale di detection ────
def run_baby_cry_detection(mother_audio_paths, stop_event_trigger, progress=gr.Progress(track_tqdm=True)):
    global shared_stop_event

    if sed_model is None or BABY_IDX == -1:
        err_msg = "ERROR: PANNs model not loaded"
        if BABY_IDX == -1:
            err_msg += " (Baby cry label not found)."
        else:
            err_msg += "."
        err_msg += " Cannot start detection."
        yield err_msg
        yield {
            status_display: gr.Textbox(value="Error: Model/Label Issue", interactive=False),
            start_button: gr.Button(value="Start Monitoring", visible=True, interactive=True),
            stop_button: gr.Button(value="Stop Monitoring", visible=False, interactive=False),
        }
        return

    if not mother_audio_paths or len(mother_audio_paths) == 0:
        yield "ERROR: Please upload at least one mother's voice MP3."
        yield {
            status_display: gr.Textbox(value="Error: No mother's voice provided", interactive=False),
            start_button: gr.Button(value="Start Monitoring", visible=True, interactive=True),
            stop_button: gr.Button(value="Stop Monitoring", visible=False, interactive=False),
        }
        return

    mother_segments = []
    for audio_path in mother_audio_paths:
        try:
            segment = AudioSegment.from_file(audio_path)
            mother_segments.append(segment)
            yield f"✓ Loaded: {os.path.basename(audio_path)}"
        except Exception as e:
            yield f"ERROR loading {os.path.basename(audio_path)}: {e}"
    
    if len(mother_segments) == 0:
        yield "ERROR: None of the uploaded audio files could be loaded."
        yield {
            status_display: gr.Textbox(value="Error: No valid audio files", interactive=False),
            start_button: gr.Button(value="Start Monitoring", visible=True, interactive=True),
            stop_button: gr.Button(value="Stop Monitoring", visible=False, interactive=False),
        }
        return
    
    yield f"Successfully loaded {len(mother_segments)} mother's voice recordings."

    shared_stop_event.clear()
    audio_queue = queue.Queue(maxsize=10) 
    log_queue = queue.Queue()

    # ── Worker detection ───
    def inference_worker():
        while not shared_stop_event.is_set():
            try:
                batch = audio_queue.get(timeout=0.1)
                if batch is None: 
                    break
                
                if batch.ndim == 1:
                    batch = batch[np.newaxis, :] # Ensure (1, num_samples)
                
                # Perform inference.
                # Based on user's error log, sed_model.inference(batch) *directly*
                # returns a 3D NumPy array (batch_size, num_frames, num_classes).
                # Or it could be a list/tuple containing this array as the first element.
                model_output_raw = sed_model.inference(batch)
                
                model_output_np = None
                if isinstance(model_output_raw, np.ndarray):
                    model_output_np = model_output_raw
                elif isinstance(model_output_raw, (list, tuple)) and model_output_raw:
                    # If it's a list/tuple, assume the framewise data is the first element
                    # This aligns with the original script's sed.inference(batch)[0]
                    if isinstance(model_output_raw[0], np.ndarray):
                        model_output_np = model_output_raw[0]
                    else:
                        log_queue.put(f"Warning: Inference result was list/tuple, but first element not np.ndarray. Type: {type(model_output_raw[0])}")
                        audio_queue.task_done()
                        continue
                else:
                    log_queue.put(f"Warning: Inference did not return np.ndarray or list/tuple. Type: {type(model_output_raw)}")
                    audio_queue.task_done()
                    continue

                if model_output_np is None or model_output_np.size == 0:
                    log_queue.put("Warning: Inference resulted in empty or invalid data.")
                    audio_queue.task_done()
                    continue
                
                # Now, model_output_np is expected to be our primary data,
                # likely shape (1, num_frames, num_classes) or already (num_frames, num_classes)
                actual_framewise_2d = None
                if model_output_np.ndim == 3:
                    if model_output_np.shape[0] == 1: # e.g., (1, 201, 527)
                        actual_framewise_2d = model_output_np[0] # Squeeze batch dim -> (201, 527)
                    else:
                        # Unlikely for our batch_size=1 input, but handle defensively
                        log_queue.put(f"Warning: Inference output 3D with batch_size > 1: {model_output_np.shape}. Taking max over batch dim.")
                        actual_framewise_2d = np.max(model_output_np, axis=0)
                elif model_output_np.ndim == 2: # Already (num_frames, num_classes)
                    actual_framewise_2d = model_output_np
                else:
                    log_queue.put(f"Warning: Unexpected inference output dimensions after processing: {model_output_np.shape}. Expected 3D or 2D array.")
                    audio_queue.task_done()
                    continue
                
                if actual_framewise_2d is None or actual_framewise_2d.ndim != 2:
                    log_queue.put(f"Error: Could not resolve framewise data to 2D. Original shape: {model_output_np.shape}")
                    audio_queue.task_done()
                    continue
                
                # --- Scoring logic using actual_framewise_2d ---
                num_frames_out, num_classes_out = actual_framewise_2d.shape
                if num_classes_out <= BABY_IDX:
                    log_queue.put(f"Warning: Framewise data has {num_classes_out} classes, insufficient for BABY_IDX ({BABY_IDX}). Shape: {actual_framewise_2d.shape}")
                    audio_queue.task_done()
                    continue

                max_probs_per_class_in_window = np.max(actual_framewise_2d, axis=0) # Shape: (num_classes,)
                
                if BABY_IDX >= len(max_probs_per_class_in_window): # Defensive check
                    log_queue.put(f"Error: BABY_IDX ({BABY_IDX}) out of bounds for class probabilities (len {len(max_probs_per_class_in_window)}).")
                    audio_queue.task_done()
                    continue
                
                score = max_probs_per_class_in_window[BABY_IDX]
                
                log_msg = f"Baby-cry score: {score:.2f}"
                if score > THRESHOLD:
                    log_msg += f"  🚨 Baby cry detected! (thresh: {THRESHOLD})"
                    log_queue.put(log_msg)
                    try:
                        selected_segment = random.choice(mother_segments)
                        play(selected_segment) 
                    except Exception as e:
                        log_queue.put(f"Error playing mother's voice: {e}")
                else:
                    log_msg += "  (no cry)"
                    log_queue.put(log_msg)
                audio_queue.task_done()

            except queue.Empty:
                continue 
            except Exception as e:
                tb_str = traceback.format_exc()
                log_queue.put(f"FATAL Error in inference worker: {e}\nTraceback:\n{tb_str}")
                shared_stop_event.set()
                break 
        log_queue.put("Inference worker stopped.")
    # ── End of corrected inference_worker ──────────────────────

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()
    yield "Inference worker started."

    pa = None
    stream = None
    try:
        pa = pyaudio.PyAudio()
        input_device_index = pa.get_default_input_device_info()['index'] 
        
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=SR,
                         input=True,
                         frames_per_buffer=HOP_SIZE,
                         input_device_index=input_device_index,
                         stream_callback=None)

        yield f"🎧 Listening... (Mic: {pa.get_device_info_by_index(input_device_index)['name']}, SR: {SR}, Win: {WINDOW_SEC}s, Hop: {HOP_SEC}s)"
        
        buffer = np.zeros((0,), dtype="float32")
        stream.start_stream()

        while not shared_stop_event.is_set():
            try:
                while not log_queue.empty():
                    yield log_queue.get_nowait()
                
                if not stream.is_active() and not shared_stop_event.is_set(): # Check if stream died unexpectedly
                    yield "ERROR: PyAudio stream is not active. Restarting might be needed."
                    shared_stop_event.set()
                    break

                raw_data = stream.read(HOP_SIZE, exception_on_overflow=False)
                mono_data = np.frombuffer(raw_data, dtype=np.int16).astype("float32") / 32768.0
                buffer = np.concatenate((buffer, mono_data))
                
                while len(buffer) >= WINDOW_SIZE:
                    if shared_stop_event.is_set(): break
                    segment_to_process = buffer[:WINDOW_SIZE].copy()
                    try:
                        audio_queue.put(segment_to_process, timeout=0.1) # Non-blocking put with timeout
                    except queue.Full:
                        log_queue.put("Warning: Audio queue full. Skipping a segment. Inference might be too slow.")
                    buffer = buffer[HOP_SIZE:] # Slide buffer
                
                time.sleep(0.01) # Brief sleep to yield control, reduce CPU in tight loop

            except IOError as e:
                if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed:
                    yield "Warning: PyAudio input overflowed. Consider increasing HOP_SIZE or reducing processing load."
                    buffer = np.zeros((0,), dtype="float32") # Clear buffer on overflow
                else:
                    yield f"PyAudio IOError: {e}"
                    shared_stop_event.set()
                    break
            except Exception as e:
                tb_str = traceback.format_exc()
                yield f"Error in main audio loop: {e}\nTraceback:\n{tb_str}"
                shared_stop_event.set()
                break
        
        yield "🔴 Stop signal received by main loop. Shutting down..."

    except Exception as e:
        tb_str = traceback.format_exc()
        yield f"CRITICAL ERROR during PyAudio setup or main loop: {e}\nTraceback:\n{tb_str}"
    finally:
        shared_stop_event.set() # Ensure stop for all threads
        
        if stream:
            try:
                if stream.is_active(): stream.stop_stream()
                stream.close()
                yield "PyAudio stream closed."
            except Exception as e_stream: yield f"Error closing stream: {e_stream}"
        if pa:
            try:
                pa.terminate()
                yield "PyAudio terminated."
            except Exception as e_pa: yield f"Error terminating PyAudio: {e_pa}"

        if 'worker' in locals() and worker.is_alive():
            log_queue.put("Main loop ending. Signaling worker to stop.")
            audio_queue.put(None) # Sentinel for worker
            worker.join(timeout=3.0) # Wait a bit longer
            if worker.is_alive():
                yield "Warning: Inference worker did not terminate cleanly after 3s."
            else:
                yield "Inference worker joined."
        
        # Drain any final logs
        while not log_queue.empty():
            try: yield log_queue.get_nowait()
            except queue.Empty: break
        
        yield "👶 Monitoring stopped. Session ended."
        yield {
            status_display: gr.Textbox(value="Idle - Session Ended", interactive=False),
            start_button: gr.Button(value="Start Monitoring", visible=True, interactive=True),
            stop_button: gr.Button(value="Stop Monitoring", visible=False, interactive=False),
        }

# ── Global state for UI control ───────────────────────────────────
shared_stop_event = threading.Event()

# ── Gradio UI Functions ───────────────────────────────────────────
def handle_start_monitoring(mother_audio_path, current_logs):
    global shared_stop_event
    shared_stop_event.clear() 

    initial_logs = (current_logs if current_logs and isinstance(current_logs, str) else "") 
    if not initial_logs.endswith("\n\n"): initial_logs += "\n" # Ensure separation
    initial_logs += "--- New Session Started ---\n"
    
    # This first yield updates the UI immediately
    yield {
        status_display: gr.Textbox(value="🚀 Initializing...", interactive=False),
        log_output: gr.Textbox(value=initial_logs, interactive=False, autoscroll=True),
        start_button: gr.Button(visible=False),
        stop_button: gr.Button(visible=True, interactive=True),
    }

    # Then, the generator continues, yielding logs from run_baby_cry_detection
    accumulated_logs = initial_logs
    for log_or_update in run_baby_cry_detection(mother_audio_path, None):
        if isinstance(log_or_update, str):
            accumulated_logs += log_or_update + "\n"
            yield {log_output: gr.Textbox(value=accumulated_logs, interactive=False, autoscroll=True)}
        elif isinstance(log_or_update, dict): # For final status updates
            if log_output in log_or_update and isinstance(log_or_update[log_output], gr.Textbox) and log_or_update[log_output].value is not None:
                 accumulated_logs = log_or_update[log_output].value 
            elif 'log_output' in log_or_update and isinstance(log_or_update['log_output'], str):
                 accumulated_logs = log_or_update['log_output']
            yield log_or_update # Pass the dict through
        time.sleep(0.01) # Allows UI to refresh smoothly

def handle_stop_monitoring():
    global shared_stop_event
    if shared_stop_event: 
        shared_stop_event.set()
    # The run_baby_cry_detection generator will handle the final state update.
    # This just signals and updates button interactivity.
    return {
        status_display: gr.Textbox(value="🛑 Sending stop signal...", interactive=False),
        stop_button: gr.Button(value="Stopping...", interactive=False)
    }

def clear_logs_func(): # Renamed to avoid conflict with variable
    return "" 

# ── Gradio UI Layout ───────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), title="Baby Cry Detector") as demo:
    gr.Markdown(
        """
        # 👶 Baby Cry Detector 🎶
        Upload one or more mother's voice MP3 files. Click "Start Monitoring" to listen via your microphone.
        When a baby cry is detected, a random mother's voice recording will play.
        Logs and scores appear below.
        """
    )
    
    model_status_msg = "Idle - Model loaded"
    if sed_model is None:
        model_status_msg = "Error - Model NOT loaded, check console!"
    if BABY_IDX == -1 and sed_model is not None: # Model loaded but label issue
        model_status_msg = "Error - Model loaded, but 'Baby cry' label NOT found!"
    elif BABY_IDX == -1 and sed_model is None: # Both failed
        model_status_msg = "Critical Error - Model & Label failed, check console!"


    with gr.Row():
        with gr.Column(scale=1):
            mother_audio_input = gr.Files(
                label="Upload Mom's Voice Files (MP3)",
                file_types=["audio"],
                file_count="multiple"
            )
            start_button = gr.Button("Start Monitoring", variant="primary", visible=True, interactive=sed_model is not None and BABY_IDX !=-1)
            stop_button = gr.Button("Stop Monitoring", variant="stop", visible=False)
            if sed_model is None or BABY_IDX == -1:
                 gr.Markdown("<p style='color:red;'>⚠️ Monitoring disabled until model/label issue is resolved. Check console logs.</p>")


        with gr.Column(scale=2):
            status_display = gr.Textbox(
                label="Status",
                value=model_status_msg,
                interactive=False,
                max_lines=1
            )
            log_output = gr.Textbox(
                label="Detection Log",
                lines=15,
                interactive=False,
                autoscroll=True,
                show_copy_button=True
            )
            clear_logs_button = gr.Button("Clear Logs")

    start_button.click(
        fn=handle_start_monitoring,
        inputs=[mother_audio_input, log_output],
        outputs=[status_display, log_output, start_button, stop_button], # Outputs must be components
        show_progress="full"
    )

    stop_button.click(
        fn=handle_stop_monitoring,
        inputs=None,
        outputs=[status_display, stop_button], # Components to update
    )
    
    clear_logs_button.click(
        fn=clear_logs_func, # Use the renamed function
        inputs=None,
        outputs=[log_output] # Component to update
    )

if __name__ == "__main__":
    if sed_model is None or BABY_IDX == -1:
        print("CRITICAL: Model not loaded or 'Baby cry' label not found. Detection will not work.")
        if not os.path.exists(WEIGHTS_PATH): print(f" - Ensure weights file exists at: {WEIGHTS_PATH}")
        if not TORCH_AVAILABLE: print(f" - PyTorch might not be installed correctly.")
        if BABY_IDX == -1 : print(f" - 'Baby cry, infant cry' not found in PANNs labels list.")
    
    demo.queue()
    demo.launch(share=True, debug=True)