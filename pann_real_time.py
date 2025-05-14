import threading, queue
import numpy as np
import pyaudio
import wave
from panns_inference import SoundEventDetection, labels
import time

# ── Params ───────────────────────────────────────────────────────────────────
SR         = 32000
WINDOW_SEC = 2.0
HOP_SEC    = 3.0
THRESHOLD  = 0.3
WINDOW_SIZE = int(WINDOW_SEC * SR)

HOP_SIZE    = int(HOP_SEC    * SR)
BABY_IDX    = labels.index("Baby cry, infant cry")


# ── Model ────────────────────────────────────────────────────────────────────
sed = SoundEventDetection(
    checkpoint_path="/Users/cesaredavidepace/Desktop/Progetti/BabyCryDetector/weights/Cnn14_DecisionLevelMax_mAP=0.385.pth",
    device="mps"
)

# ── Prepare output WAV ───────────────────────────────────────────────────────
wf = wave.open("recording.wav", "wb")
wf.setnchannels(1)
wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
wf.setframerate(SR)

# ── Thread‐safe queue & stop event ───────────────────────────────────────────
audio_queue = queue.Queue()
stop_event = threading.Event()

def inference_worker():
    while not stop_event.is_set():
        batch = audio_queue.get()
        if batch is None:
            break

        # start timing
        t0 = time.time()
        framewise = sed.inference(batch)[0]
        dt = time.time() - t0

        score = np.max(framewise, axis=0)[BABY_IDX]
        # print inference time and result
        print(f"Inference time: {dt:.3f}s — baby‐cry score: {score:.2f}", end='')
        if score > THRESHOLD:
            print("  \033[91m🚨 Baby cry detected!\033[0m")
        else:
            print("  no cry")

        audio_queue.task_done()

def stop_listener():
    input("► Press ENTER to stop listening and save WAV ◄\n")
    stop_event.set()
    audio_queue.put(None)

# start threads
worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()
threading.Thread(target=stop_listener, daemon=True).start()

# ── PyAudio capture ──────────────────────────────────────────────────────────
pa = pyaudio.PyAudio()
stream = pa.open(format            = pyaudio.paInt16,
                 channels          = 1,
                 rate              = SR,
                 input             = True,
                 frames_per_buffer = HOP_SIZE)

buffer = np.zeros((0,), dtype="float32")
print("Listening for baby cries… (hit ENTER to stop)")

try:
    while not stop_event.is_set():
        # 1) Read block
        raw = stream.read(HOP_SIZE, exception_on_overflow=False)

        # 2) Write raw bytes into WAV file
        wf.writeframes(raw)

        # 3) Convert to float32 for detection
        mono = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0

        # 4) Buffer / sliding window logic
        buffer = np.concatenate((buffer, mono))
        while len(buffer) >= WINDOW_SIZE:
            segment = buffer[:WINDOW_SIZE]
            audio_queue.put(segment[np.newaxis, :])
            buffer = buffer[HOP_SIZE:]
finally:
    # clean up everything
    stop_event.set()
    audio_queue.put(None)
    worker.join()

    stream.stop_stream()
    stream.close()
    pa.terminate()

    wf.close()    # ensure WAV is properly finalized
    print("Recording saved to recording.wav. Exited cleanly.")