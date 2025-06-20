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
    "last_ui_state": None,  # Memorizza lo stato di rilevamento (True/False)
    "mother_segments": [],  # Lista di tuple (nome_file, AudioSegment)
    "monitoring_active": False,  # Flag per lo stato del monitoraggio
    "current_score": 0.0,  # Score corrente per evitare aggiornamenti non necessari
    "last_status_text": "",  # Ultimo testo di stato per evitare aggiornamenti
    "last_visual_html": ""  # Ultimo HTML visivo per evitare aggiornamenti
}

# ── Modello (caricato una sola volta) ──────────────────────────────
print("Caricamento modello di sound detection...")
sed = SoundEventDetection(
    checkpoint_path="weights/Cnn14_DecisionLevelMax_mAP=0.385.pth",
    device="cuda"
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
                file_name, selected_segment = random.choice(mother_segments)
                print(f"PIANTO RILEVATO (Score: {score:.2f}) - Riproduco: {file_name}")
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
def add_audio_file(file_paths, current_files_text):
    """Aggiunge uno o più file audio alla lista."""
    if not file_paths:
        return current_files_text, "⚠️ Nessun file selezionato."

    added_count = 0
    skipped_count = 0
    error_list = []

    for file_path in file_paths:
        try:
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path.split("\\")[-1]

            # Evita duplicati basati sul nome del file
            if any(name == file_name for name, seg in app_state["mother_segments"]):
                print(f"File '{file_name}' già presente, saltato.")
                skipped_count += 1
                continue

            segment = AudioSegment.from_file(file_path)
            app_state["mother_segments"].append((file_name, segment))
            added_count += 1

        except Exception as e:
            error_list.append(f"{file_path}: {e}")

    # Aggiorna la lista mostrata nell'interfaccia
    file_names = [name for name, seg in app_state["mother_segments"]]
    updated_files_text = "\n".join([f"• {name}" for name in file_names]) or "Nessun file caricato"

    # Crea messaggio di stato
    status_parts = []
    if added_count > 0:
        status_parts.append(f"✅ Aggiunti {added_count} file.")
    if skipped_count > 0:
        status_parts.append(f"ℹ️ Saltati {skipped_count} duplicati.")
    if error_list:
        status_parts.append(f"❌ {len(error_list)} errori.")
    
    if not status_parts:
         status_message = "ℹ️ Nessun nuovo file aggiunto (potrebbero essere duplicati o errati)."
    else:
        status_message = " ".join(status_parts)
    
    status_message += f" Totale: {len(app_state['mother_segments'])} file."

    return updated_files_text, status_message


def clear_audio_files():
    """Pulisce tutti i file audio caricati."""
    app_state["mother_segments"].clear()
    return "Nessun file caricato", "🗑️ Tutti i file sono stati rimossi."


# ── Funzioni per l'interfaccia Gradio ────────────────────────────
def start_detection():
    """Funzione chiamata dal pulsante 'Avvia'."""
    if app_state.get("monitoring_active"):
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

    # Imposta lo stato iniziale per l'UI
    app_state["monitoring_active"] = True
    app_state["last_ui_state"] = False  # Inizializza a non rilevato

    return (
        f"✅ Monitoraggio avviato! In ascolto con {len(app_state['mother_segments'])} file(s) audio...",
        create_visual_status("🟢 In Ascolto", "listening"),
        create_score_display(0.0)
    )


def stop_detection():
    """Funzione chiamata dal pulsante 'Ferma'."""
    if app_state.get("monitoring_active"):
        app_state["stop_event"].set()

        # Attendi la terminazione dei thread
        if app_state["pyaudio_thread"]: app_state["pyaudio_thread"].join(timeout=2)
        if app_state["inference_thread"]: app_state["inference_thread"].join(timeout=2)

        # Resetta lo stato globale
        app_state["monitoring_active"] = False
        app_state["stop_event"] = None
        app_state["inference_thread"] = None
        app_state["pyaudio_thread"] = None
        app_state["last_ui_state"] = None

        return (
            "🔴 Monitoraggio fermato.",
            create_visual_status("⚫ Sistema Spento", "offline"),
            create_score_display(0.0)
        )

    # Se non era attivo, restituisce lo stato di default
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
    if not app_state.get("monitoring_active"):
        return gr.skip(), gr.skip(), gr.skip()

    # Controlla se ci sono nuovi risultati nella coda
    new_results = []
    try:
        # Raccoglie tutti i risultati disponibili per processare solo l'ultimo
        while True:
            result = result_queue.get_nowait()
            new_results.append(result)
    except queue.Empty:
        pass

    # Se non ci sono nuovi risultati, non aggiornare nulla
    if not new_results:
        return gr.skip(), gr.skip(), gr.skip()

    # Prendi solo l'ultimo risultato per evitare lag
    result = new_results[-1]
    score = result['score']
    detected = result['detected']

    # Controlla se il punteggio è cambiato significativamente
    score_changed = abs(score - app_state["current_score"]) > 0.001
    
    # Ottieni lo stato di rilevamento precedente
    last_detected = app_state["last_ui_state"]
    detection_state_changed = detected != last_detected

    # Se nè lo stato nè il punteggio sono cambiati, non aggiornare nulla
    if not detection_state_changed and not score_changed:
        return gr.skip(), gr.skip(), gr.skip()

    # Prepara i nuovi contenuti
    if detected:
        new_status_text = "✅ Monitoraggio attivo - PIANTO RILEVATO!"
        new_visual_html = create_visual_status("🚨 PIANTO RILEVATO!", "cry-alert")
    else:
        new_status_text = "✅ Monitoraggio attivo - In ascolto..."
        new_visual_html = create_visual_status("🟢 In Ascolto", "listening")

    new_score_html = create_score_display(score)

    # Determina cosa aggiornare
    status_to_return = new_status_text if detection_state_changed else gr.skip()
    visual_to_return = new_visual_html if detection_state_changed else gr.skip()
    score_to_return = new_score_html if score_changed else gr.skip()

    # Aggiorna lo stato interno solo se necessario
    if detection_state_changed:
        app_state["last_ui_state"] = detected
        app_state["last_status_text"] = new_status_text
        app_state["last_visual_html"] = new_visual_html
    
    if score_changed:
        app_state["current_score"] = score

    return status_to_return, visual_to_return, score_to_return


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
            mother_audio_input = gr.File(
                label="Seleziona uno o più file audio da aggiungere",
                type="filepath",
                file_count="multiple",
                file_types=["audio"]
            )

            with gr.Row():
                add_button = gr.Button("➕ Aggiungi File/i", variant="primary")
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
    timer = gr.Timer(value=1.0)  # Ridotto a 1 secondo per maggiore reattività
    timer.tick(
        fn=update_ui_complete,
        outputs=[system_status, visual_status, score_display]
    )

# Avvia l'interfaccia
if __name__ == "__main__":
    demo.launch(share=True)
