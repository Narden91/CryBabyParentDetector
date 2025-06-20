import gradio as gr
import threading
import queue
import numpy as np
import pyaudio
import wave
import time
import random
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
    "pyaudio_thread": None,
    "last_ui_state": None,  # Per evitare aggiornamenti non necessari
    "mother_segments": []  # Lista di file audio caricati
}

# ── Modello (caricato una sola volta) ──────────────────────────────
print("Caricamento modello di sound detection...")
sed = SoundEventDetection(
    checkpoint_path="weights/Cnn14_DecisionLevelMax_mAP=0.385.pth",
    device="cpu"
)
print("Modello caricato.")


# ── Worker per l'inferenza ───────────────
def inference_worker(mother_segments, stop_event):
    """Esegue l'inferenza sui dati audio e mette i risultati in una coda."""
    while not stop_event.is_set():
        try:
            batch = audio_queue.get(timeout=1)
            if batch is None: break

            framewise = sed.inference(batch)[0]
            score = np.max(framewise, axis=0)[BABY_IDX]

            detected = score > THRESHOLD
            result_queue.put({"score": score, "detected": detected})

            if detected and mother_segments:
                # Seleziona casualmente uno dei file audio caricati
                selected_segment = random.choice(mother_segments)
                print(f"PIANTO RILEVATO (Score: {score:.2f}) - Riproduco audio casuale.")
                play(selected_segment)

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


# ── Funzioni per la gestione dei file multipli ────────────────────────────
def add_audio_file(file_path, current_files):
    """Aggiunge un file audio alla lista."""
    if file_path is None:
        return current_files, "⚠️ Nessun file selezionato."

    try:
        # Carica il file per verificare che sia valido
        segment = AudioSegment.from_file(file_path)

        # Aggiungi alla lista globale
        app_state["mother_segments"].append(segment)

        # Aggiorna la lista mostrata nell'interfaccia
        file_name = file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1]

        # Crea la lista completa dei file
        if len(app_state["mother_segments"]) == 1:
            updated_files = f"• {file_name}"
        else:
            # Ricostruisci la lista con tutti i file (semplificato per il debug)
            updated_files = "\n".join([f"• File {i + 1}" for i in range(len(app_state["mother_segments"]))])

        return updated_files, f"✅ File '{file_name}' aggiunto con successo! Totale: {len(app_state['mother_segments'])} file(s)."

    except Exception as e:
        return current_files, f"❌ Errore nel caricamento del file: {str(e)}"


def clear_audio_files():
    """Pulisce tutti i file audio caricati."""
    app_state["mother_segments"].clear()
    return "Nessun file caricato", "🗑️ Tutti i file sono stati rimossi."


# ── Funzioni per l'interfaccia Gradio ────────────────────────────
def start_detection():
    """Funzione chiamata dal pulsante 'Avvia'."""
    if app_state["inference_thread"] is not None and app_state["inference_thread"].is_alive():
        return "Il monitoraggio è già attivo.", create_visual_status("🟢 In Ascolto", "listening"), create_score_display(
            0.0)

    if not app_state["mother_segments"]:
        raise gr.Error("Per favore, carica almeno un file audio prima di iniziare.")

    # Pulisci le code prima di iniziare
    while not audio_queue.empty(): audio_queue.get()
    while not result_queue.empty(): result_queue.get()

    app_state["stop_event"] = threading.Event()

    # Avvia il thread di inferenza
    app_state["inference_thread"] = threading.Thread(
        target=inference_worker,
        args=(app_state["mother_segments"], app_state["stop_event"]),
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

    # Reset dello stato UI
    app_state["last_ui_state"] = None

    return (
        f"✅ Monitoraggio avviato! In ascolto con {len(app_state['mother_segments'])} file(s) audio...",
        create_visual_status("🟢 In Ascolto", "listening"),
        create_score_display(0.0)
    )


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
        app_state["last_ui_state"] = None

        return (
            "🔴 Monitoraggio fermato.",
            create_visual_status("⚫ Sistema Spento", "offline"),
            create_score_display(0.0)
        )
    return (
        "Il monitoraggio non era attivo.",
        create_visual_status("⚫ Sistema Spento", "offline"),
        create_score_display(0.0)
    )


# ── Funzioni per creare l'HTML dei componenti ─────────────────────
def create_visual_status(text, css_class):
    """Crea l'HTML per lo stato visivo."""
    return f"""
    <div class="big-status {css_class}">
        {text}
    </div>
    """


def create_score_display(score):
    """Crea l'HTML per il display del punteggio."""
    return f"""
    <div class="score-display">
        📊 Punteggio: {score:.3f} / {THRESHOLD}
    </div>
    """


# ── Aggiornamento UI ottimizzato per evitare il blinking ─────────────────────
def update_ui_complete():
    """Funzione per aggiornare l'UI in tempo reale solo quando necessario."""
    # Se il monitoraggio non è attivo, non fare nulla
    if app_state["stop_event"] is None or not app_state["inference_thread"] or not app_state[
        "inference_thread"].is_alive():
        return gr.skip(), gr.skip(), gr.skip()

    try:
        result = result_queue.get_nowait()
        score = result['score']
        detected = result['detected']

        # Crea il nuovo stato
        if detected:
            status_text = "✅ Monitoraggio attivo - PIANTO RILEVATO!"
            visual_html = create_visual_status("🚨 PIANTO RILEVATO!", "cry-alert")
        else:
            status_text = "✅ Monitoraggio attivo - In ascolto..."
            visual_html = create_visual_status("🟢 In Ascolto", "listening")

        score_html = create_score_display(score)

        # Controlla se lo stato è cambiato per evitare aggiornamenti non necessari
        new_state = (status_text, detected, round(score, 3))
        if app_state["last_ui_state"] == new_state:
            return gr.skip(), gr.skip(), gr.skip()

        app_state["last_ui_state"] = new_state
        return status_text, visual_html, score_html

    except queue.Empty:
        # Nessun nuovo risultato disponibile
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

.file-list {
    background: rgba(240, 248, 255, 0.8) !important;
    border: 1px solid #87ceeb !important;
    border-radius: 8px !important;
    padding: 10px !important;
    font-family: monospace !important;
    white-space: pre-line !important;
    max-height: 150px !important;
    overflow-y: auto !important;
}
"""

# ── Interfaccia Gradio con elementi visivi migliorati ─────────────────────
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 👶 Baby Cry Detector 🎶 (Multi-File Edition)
        **Soglia di rilevamento fissa: 0.2**

        **Novità:**
        - ✨ **Supporto per file multipli**: Carica più file audio e il sistema ne sceglierà uno casualmente quando rileva il pianto
        - 🚀 **Interface ottimizzata**: Niente più blinking durante gli aggiornamenti

        **Come funziona:**
        1. Carica uno o più file audio (MP3, WAV) con voci o melodie tranquillizzanti
        2. Clicca su **Avvia Monitoraggio** per iniziare l'ascolto
        3. Quando viene rilevato il pianto, il sistema riprodurrà casualmente uno dei file caricati!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎵 Gestione File Audio")

            # Sezione per aggiungere file
            mother_audio_input = gr.Audio(
                label="Seleziona File Audio da Aggiungere",
                type="filepath"
            )

            with gr.Row():
                add_button = gr.Button("➕ Aggiungi File", variant="primary")
                clear_button = gr.Button("🗑️ Rimuovi Tutti", variant="secondary")

            # Lista dei file caricati
            file_list_display = gr.Textbox(
                label="📁 File Caricati",
                value="Nessun file caricato",
                interactive=False,
                lines=4,
                elem_classes=["file-list"]
            )

            file_status = gr.Textbox(
                label="📋 Stato File",
                value="Pronto per aggiungere file...",
                interactive=False,
                lines=2
            )

            gr.Markdown("### 🎯 Controlli Monitoraggio")

            with gr.Row():
                start_button = gr.Button("▶️ Avvia Monitoraggio", variant="primary", size="lg")
                stop_button = gr.Button("⏹️ Ferma Monitoraggio", variant="stop", size="lg")

            system_status = gr.Textbox(
                label="📋 Stato del Sistema",
                value="Pronto per iniziare...",
                interactive=False,
                lines=2
            )

        with gr.Column(scale=2):
            gr.Markdown("## 🎯 Monitoraggio in Tempo Reale")

            # Grande display dello stato visivo
            visual_status = gr.HTML(
                value=create_visual_status("⚫ Sistema Spento", "offline"),
                label="Stato Visivo"
            )

            # Display del punteggio
            score_display = gr.HTML(
                value=create_score_display(0.0),
                label="Punteggio Rilevamento"
            )

    # ── Logica di interazione dei componenti ─────────────────────

    # Gestione file
    add_button.click(
        fn=add_audio_file,
        inputs=[mother_audio_input, file_list_display],
        outputs=[file_list_display, file_status]
    )

    clear_button.click(
        fn=clear_audio_files,
        inputs=None,
        outputs=[file_list_display, file_status]
    )

    # Controlli monitoraggio
    start_button.click(
        fn=start_detection,
        inputs=None,
        outputs=[system_status, visual_status, score_display]
    )

    stop_button.click(
        fn=stop_detection,
        inputs=None,
        outputs=[system_status, visual_status, score_display]
    )

    # Timer per aggiornamenti UI ottimizzati (solo quando necessario)
    timer = gr.Timer(value=2.0)  # ogni 2 secondi e SOLO quando il monitoraggio è attivo
    timer.tick(
        fn=update_ui_complete,
        outputs=[system_status, visual_status, score_display]
    )

# Avvia l'interfaccia
if __name__ == "__main__":
    demo.launch(share=True)
