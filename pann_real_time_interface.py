import gradio as gr
import threading
import queue
import numpy as np
import pyaudio
import wave
import time
from panns_inference import SoundEventDetection, labels
from pydub import AudioSegment
from pydub.playback import play
import io

# ── Parametri audio ─────────────────────────────────────────────
SR = 32000
WINDOW_SEC = 2.0
HOP_SEC = 3.0
THRESHOLD = 0.2
WINDOW_SIZE = int(WINDOW_SEC * SR)
HOP_SIZE = int(HOP_SEC * SR)
BABY_IDX = labels.index("Baby cry, infant cry")

# ── Modello ─────────────────────────────────────────────────────
sed = SoundEventDetection(
    checkpoint_path="weights/Cnn14_DecisionLevelMax_mAP=0.385.pth", 
    device="cuda"  
)


# ── Funzione principale ─────────────────────────────────────────
def start_detection(mother_audio_path):
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # Convert MP3 in AudioSegment
    mother_segment = AudioSegment.from_file(mother_audio_path)

    # ── Worker detection ────────────────────────────────────────
    def inference_worker():
        while not stop_event.is_set():
            batch = audio_queue.get()
            if batch is None:
                break
            framewise = sed.inference(batch)[0]
            score = np.max(framewise, axis=0)[BABY_IDX]
            print(f"baby‐cry score: {score:.2f}", end='')
            if score > THRESHOLD:
                print("  🚨 Baby cry detected!")
                play(mother_segment)
            else:
                print("  no cry")
            audio_queue.task_done()

    # ── Avvia il worker thread ──────────────────────────────────
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    # ── PyAudio stream ──────────────────────────────────────────
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=SR,
                     input=True,
                     frames_per_buffer=HOP_SIZE)

    buffer = np.zeros((0,), dtype="float32")
    print("🎧 Listening for baby cries… (Ctrl+C to stop)")

    try:
        while not stop_event.is_set():
            raw = stream.read(HOP_SIZE, exception_on_overflow=False)
            mono = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
            buffer = np.concatenate((buffer, mono))
            while len(buffer) >= WINDOW_SIZE:
                segment = buffer[:WINDOW_SIZE]
                audio_queue.put(segment[np.newaxis, :])
                buffer = buffer[HOP_SIZE:]
    except KeyboardInterrupt:
        print("🔴 Stopping detection...")
    finally:


        stop_event.set()
        audio_queue.put(None)
        worker.join()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        return "👶 Monitoring stopped. Baby cry detection session ended."


# ── Gradio UI ───────────────────────────────────────────────────
gr.Interface(
    fn=start_detection,
    inputs=[
        gr.Audio(label="Upload Mom's Voice (MP3)", type="filepath")
    ],
    outputs="text",
    title="Baby Cry Detector",
    description="Upload a synthetic mother's voice (mp3). When a baby cry is detected, the mom's voice plays."
).launch(share=True)