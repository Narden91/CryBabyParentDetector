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

# ── Parametri Audio ──────────────────────────────────
SR = 32000
WINDOW_SEC = 2.0
HOP_SEC = 1.0
THRESHOLD = 0.2  # Soglia fissa come richiesto
WINDOW_SIZE = int(WINDOW_SEC * SR)
HOP_SIZE = int(HOP_SEC * SR)
BABY_IDX = labels.index("Baby cry, infant cry")

# ── Code Globale per la gestione dello stato ───────────────────────
audio_queue = queue.Queue()
result_queue = queue.Queue()
app_state = {
    "stop_event": None,
    "inference_thread": None,
    "pyaudio_thread": None
}

# ── Modello (caricato una sola volta) ──────────────────────────────
print("Caricamento modello di sound detection...")
sed = SoundEventDetection(
    checkpoint_path="weights/Cnn14_DecisionLevelMax_mAP=0.385.pth",
    device="cpu"
)
print("Modello caricato.")


# ── Worker per l'inferenza ───────────────
def inference_worker(mother_segment, stop_event):
    """Esegue l'inferenza sui dati audio e mette i risultati in una coda."""
    while not stop_event.is_set():
        try:
            batch = audio_queue.get(timeout=1)
            if batch is None: break

            framewise = sed.inference(batch)[0]
            score = np.max(framewise, axis=0)[BABY_IDX]

            detected = score > THRESHOLD
            result_queue.put({"score": score, "detected": detected})

            if detected:
                print(f"PIANTO RILEVATO (Score: {score:.2f}) - Riproduco audio.")
                play(mother_segment)

            audio_queue.task_done()
        except queue.Empty:
            continue


# ── Worker per la cattura audio ───────────────────────────
def pyaudio_worker(stop_event):
    """Cattura l'audio dal microfono e lo mette nella coda audio."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SR, input=True, frames_per_buffer=HOP_SIZE)

    print("🎧 In ascolto...")
    buffer = np.zeros((0,), dtype="float32")

    while not stop_event.is_set():
        raw = stream.read(HOP_SIZE, exception_on_overflow=False)
        mono = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
        buffer = np.concatenate((buffer, mono))
        while len(buffer) >= WINDOW_SIZE:
            segment = buffer[:WINDOW_SIZE]
            audio_queue.put(segment[np.newaxis, :])
            buffer = buffer[HOP_SIZE:]

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("🔴 Ascolto terminato.")


# ── Funzioni per l'interfaccia Gradio ────────────────────────────
def start_detection(mother_audio_path):
    """Funzione chiamata dal pulsante 'Avvia'."""
    if app_state["inference_thread"] is not None and app_state["inference_thread"].is_alive():
        return "Il monitoraggio è già attivo.", "🟢 In Ascolto", "rgba(34, 139, 34, 0.1)", 0.0

    if mother_audio_path is None:
        raise gr.Error("Per favore, carica prima il file audio della mamma.")

    # Pulisci le code prima di iniziare
    while not audio_queue.empty(): audio_queue.get()
    while not result_queue.empty(): result_queue.get()

    mother_segment = AudioSegment.from_file(mother_audio_path)

    app_state["stop_event"] = threading.Event()

    # Avvia il thread di inferenza
    app_state["inference_thread"] = threading.Thread(
        target=inference_worker,
        args=(mother_segment, app_state["stop_event"]),
        daemon=True
    )
    app_state["inference_thread"].start()

    # Avvia il thread di cattura audio
    app_state["pyaudio_thread"] = threading.Thread(
        target=pyaudio_worker,
        args=(app_state["stop_event"],),
        daemon=True
    )
    app_state["pyaudio_thread"].start()

    return "✅ Monitoraggio avviato! In ascolto...", "🟢 In Ascolto", "rgba(34, 139, 34, 0.1)", 0.0


def stop_detection():
    """Funzione chiamata dal pulsante 'Ferma'."""
    if app_state["stop_event"]:
        app_state["stop_event"].set()

        # Attendi la terminazione dei thread
        if app_state["pyaudio_thread"]: app_state["pyaudio_thread"].join(timeout=2)
        if app_state["inference_thread"]: app_state["inference_thread"].join(timeout=2)

        app_state["stop_event"] = None
        app_state["inference_thread"] = None
        app_state["pyaudio_thread"] = None

        return "🔴 Monitoraggio fermato.", "⚫ Spento", "rgba(128, 128, 128, 0.1)", 0.0
    return "Il monitoraggio non era attivo.", "⚫ Spento", "rgba(128, 128, 128, 0.1)", 0.0


def update_ui():
    """Funzione per aggiornare l'UI in tempo reale."""
    try:
        result = result_queue.get_nowait()
        score = result['score']

        if result['detected']:
            status_text = "🚨 PIANTO RILEVATO!"
            bg_color = "rgba(255, 0, 0, 0.2)"  # Rosso con trasparenza
            return status_text, bg_color, score
        else:
            status_text = "🟢 In Ascolto"
            bg_color = "rgba(34, 139, 34, 0.1)"  # Verde con trasparenza
            return status_text, bg_color, score

    except queue.Empty:
        # Se non ci sono nuovi risultati, mantieni lo stato attuale
        return gr.skip(), gr.skip(), gr.skip()


# ── CSS personalizzato per l'effetto visivo ─────────────────────
custom_css = """
.cry-alert {
    background: linear-gradient(45deg, #ff6b6b, #ff8e8e, #ff6b6b) !important;
    animation: pulse-red 1s infinite;
    border: 3px solid #ff0000 !important;
    border-radius: 15px !important;
    box-shadow: 0 0 20px rgba(255, 0, 0, 0.5) !important;
}

.listening {
    background: linear-gradient(45deg, #51cf66, #69db7c, #51cf66) !important;
    animation: pulse-green 2s infinite;
    border: 2px solid #228b22 !important;
    border-radius: 15px !important;
    box-shadow: 0 0 15px rgba(34, 139, 34, 0.3) !important;
}

.offline {
    background: rgba(128, 128, 128, 0.1) !important;
    border: 2px solid #808080 !important;
    border-radius: 15px !important;
}

@keyframes pulse-red {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.05); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes pulse-green {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
}

.big-status {
    font-size: 2.5em !important;
    font-weight: bold !important;
    text-align: center !important;
    padding: 30px !important;
    margin: 20px 0 !important;
    min-height: 120px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.score-display {
    font-size: 1.5em !important;
    font-weight: bold !important;
    text-align: center !important;
    padding: 15px !important;
    border-radius: 10px !important;
    background: rgba(240, 240, 240, 0.8) !important;
}
"""

# ── Interfaccia Gradio con elementi visivi migliorati ─────────────────────
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 👶 Baby Cry Detector 🎶
        **Soglia di rilevamento fissa: 0.2**

        **Come funziona:**
        1. Carica un file audio (MP3, WAV) con una voce o una melodia tranquillizzante
        2. Clicca su **Avvia Monitoraggio** per iniziare l'ascolto
        3. Quando viene rilevato il pianto, vedrai un grande allarme rosso lampeggiante!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            mother_audio_input = gr.Audio(
                label="🎵 Carica Voce della Mamma (o melodia)",
                type="filepath"
            )
            with gr.Row():
                start_button = gr.Button("▶️ Avvia Monitoraggio", variant="primary", size="lg")
                stop_button = gr.Button("⏹️ Ferma Monitoraggio", variant="stop", size="lg")

            status_box = gr.Textbox(
                label="📋 Stato del Sistema",
                value="Pronto per iniziare...",
                interactive=False,
                lines=2
            )

        with gr.Column(scale=2):
            gr.Markdown("## 🎯 Monitoraggio in Tempo Reale")

            # Grande display dello stato visivo
            visual_status = gr.HTML(
                value="""
                <div class="big-status offline">
                    ⚫ Sistema Spento
                </div>
                """,
                label="Stato Visivo"
            )

            # Display del punteggio
            score_display = gr.HTML(
                value="""
                <div class="score-display">
                    📊 Punteggio: 0.00
                </div>
                """,
                label="Punteggio Rilevamento"
            )


    # Funzione per aggiornare l'HTML dello stato visivo
    def update_visual_status(status_text, bg_color, score):
        if "PIANTO RILEVATO" in status_text:
            css_class = "cry-alert"
        elif "In Ascolto" in status_text:
            css_class = "listening"
        else:
            css_class = "offline"

        visual_html = f"""
        <div class="big-status {css_class}">
            {status_text}
        </div>
        """

        score_html = f"""
        <div class="score-display">
            📊 Punteggio: {score:.3f} / {THRESHOLD}
        </div>
        """

        return visual_html, score_html


    # Funzioni semplificate per i pulsanti
    def start_ui_update(mother_audio_path):
        status, visual_text, _, score = start_detection(mother_audio_path)
        visual_html, score_html = update_visual_status(visual_text, "", score)
        return status, visual_html, score_html


    def stop_ui_update():
        status, visual_text, _, score = stop_detection()
        visual_html, score_html = update_visual_status(visual_text, "", score)
        return status, visual_html, score_html


    # Aggiornamento UI in tempo reale
    def update_ui_complete():
        try:
            result = result_queue.get_nowait()
            score = result['score']

            if result['detected']:
                status_text = "✅ Monitoraggio attivo - PIANTO RILEVATO!"
                visual_text = "🚨 PIANTO RILEVATO!"
                visual_html, score_html = update_visual_status(visual_text, "", score)
                return status_text, visual_html, score_html
            else:
                status_text = "✅ Monitoraggio attivo - In ascolto..."
                visual_text = "🟢 In Ascolto"
                visual_html, score_html = update_visual_status(visual_text, "", score)
                return status_text, visual_html, score_html

        except queue.Empty:
            return gr.skip(), gr.skip(), gr.skip()


    # Logica di interazione dei componenti
    start_button.click(
        fn=start_ui_update,
        inputs=[mother_audio_input],
        outputs=[status_box, visual_status, score_display]
    )

    stop_button.click(
        fn=stop_ui_update,
        inputs=None,
        outputs=[status_box, visual_status, score_display]
    )

    # ───> qui sostituiamo `demo.load(every=0.5)` con un Timer <───
    timer = gr.Timer(value=0.5)  # ogni 0.5 secondi
    timer.tick(
        fn=update_ui_complete,
        outputs=[status_box, visual_status, score_display]
    )

# Avvia l'interfaccia
if __name__ == "__main__":
    demo.launch(share=True)
