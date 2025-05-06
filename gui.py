import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Imposta il backend Qt5 per compatibilità con i thread
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QSlider, QFileDialog, 
    QGroupBox, QRadioButton, QCheckBox, QListWidget, QProgressBar,
    QTabWidget, QSplitter, QSpacerItem, QSizePolicy, QTextEdit,
    QButtonGroup, QScrollArea, QListWidgetItem, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QFont, QColor, QPalette

# Importa i moduli dall'applicazione esistente
from core.detector import BabyCryDetector
from utils.helpers import process_uploaded_files, format_file_list

# Percorso predefinito del modello
DEFAULT_MODEL_PATH = os.path.join("models", "baby_cry_svm.joblib")

# Stili e temi per l'UI
STYLE = """
QMainWindow {
    background-color: #f5f5f7;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #cccccc;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}
QPushButton {
    background-color: #5b88a5;
    color: white;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6b98b5;
}
QPushButton:pressed {
    background-color: #4b7895;
}
QPushButton#startButton {
    background-color: #4CAF50;
}
QPushButton#startButton:hover {
    background-color: #5CBF60;
}
QPushButton#stopButton {
    background-color: #F44336;
}
QPushButton#stopButton:hover {
    background-color: #FF5346;
}
QComboBox, QListWidget {
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 4px;
    background-color: white;
}
QSlider::groove:horizontal {
    border: 1px solid #999999;
    height: 8px;
    background: #cccccc;
    margin: 2px 0;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #5b88a5;
    border: 1px solid #5b88a5;
    width: 18px;
    margin: -2px 0;
    border-radius: 4px;
}
QFrame#statusFrame {
    background-color: #ffffff;
    border-radius: 6px;
    border: 1px solid #cccccc;
}
QLabel#statusLabel {
    font-weight: bold;
    color: #333333;
}
QProgressBar {
    border: 1px solid #cccccc;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #5b88a5;
    width: 20px;
}
QTextEdit {
    background-color: white;
    border: 1px solid #cccccc;
    border-radius: 4px;
}
"""

class AudioPlotCanvas(FigureCanvas):
    """Canvas per visualizzare l'audio in tempo reale"""
    
    def __init__(self, parent=None, width=6, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        # Configura il grafico
        self.axes.set_ylim(-1, 1)
        self.axes.set_xlim(0, 1000)
        self.axes.set_xlabel("Sample")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True, alpha=0.3)
        
        # Dati iniziali
        self.data = np.zeros(1000)
        self.line, = self.axes.plot(np.arange(1000), self.data, color='blue', alpha=0.7)
        self.level_rect = self.axes.add_patch(plt.Rectangle((0, -0.95), 0, 0.1, color='green', alpha=0.7))
        self.threshold_line = None
        
        # Testo per le informazioni
        self.level_text = self.axes.text(50, -0.9, "Level: 0.00", color='black', fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.7))
        self.mic_text = self.axes.text(0.02, 0.95, "Mic: Unknown", transform=self.axes.transAxes,
                             fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        self.files_text = self.axes.text(0.02, 0.90, "Audio files: 0", transform=self.axes.transAxes,
                              fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        
        self.fig.tight_layout()
    
    def update_plot(self, audio_data, level, threshold=None, mic_name="Unknown", 
                   num_files=0, playback_mode="random", is_crying=False, 
                   detection_method="simple"):
        """Aggiorna il grafico con nuovi dati audio"""
        # Aggiorna la forma d'onda
        self.line.set_ydata(audio_data)
        
        # Aggiorna indicatore del livello
        if level < 0.3:
            level_color = 'green'
        elif level < 0.7:
            level_color = 'orange'
        else:
            level_color = 'red'
            
        # Aggiorna il rettangolo del livello
        self.level_rect.set_width(1000 * level)
        self.level_rect.set_color(level_color)
        
        # Aggiorna il testo del livello
        self.level_text.set_text(f"Level: {level:.2f}")
        
        # Aggiorna le informazioni sul microfono e i file
        self.mic_text.set_text(f"Mic: {mic_name} | Mode: {detection_method}")
        self.files_text.set_text(f"Audio files: {num_files} ({playback_mode} mode)")
        
        # Aggiorna la linea di soglia se è Simple Detection
        if detection_method == "Simple level detection" and threshold is not None:
            if self.threshold_line is None:
                self.threshold_line = self.axes.axhline(y=threshold/50, color='red', 
                                                       linestyle='--', alpha=0.7)
            else:
                self.threshold_line.set_ydata([threshold/50, threshold/50])
        else:
            if self.threshold_line is not None:
                self.threshold_line.remove()
                self.threshold_line = None
        
        # Evidenzia quando viene rilevato il pianto
        if is_crying:
            self.fig.patch.set_facecolor((1.0, 0.9, 0.9))
            
            # Aggiungi testo di rilevamento se non presente
            if not hasattr(self, 'cry_text'):
                self.cry_text = self.axes.text(0.5, 0.5, 'Crying detected!', 
                                              horizontalalignment='center',
                                              verticalalignment='center', 
                                              transform=self.axes.transAxes,
                                              color='red', fontsize=12, 
                                              bbox=dict(facecolor='white', alpha=0.8))
        else:
            self.fig.patch.set_facecolor((1, 1, 1))
            
            # Rimuovi il testo di rilevamento se presente
            if hasattr(self, 'cry_text'):
                self.cry_text.remove()
                delattr(self, 'cry_text')
        
        # Ridisegna il grafico
        self.draw()

class DetectorThread(QThread):
    """Thread separato per l'esecuzione del rilevatore"""
    
    status_update = pyqtSignal(dict)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = False
    
    def run(self):
        self.running = True
        while self.running:
            # Ottieni lo stato attuale
            status = self.detector.get_status_values()
            
            # Rimuovi la figura matplotlib dallo stato per evitare warning
            if 'audio_viz' in status:
                del status['audio_viz']
            
            # Emetti il segnale con lo stato
            self.status_update.emit(status)
            time.sleep(0.2)
    
    def stop(self):
        self.running = False

class BabyCryDetectorGUI(QMainWindow):
    """Interfaccia grafica principale per il Baby Cry Detector"""
    
    def __init__(self):
        super().__init__()
        self.detector = BabyCryDetector()
        self.detector_thread = None
        
        self.initUI()
        
        # Timer per aggiornare l'interfaccia
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(1000)  # Aggiorna ogni secondo
        
        # Flag per il rilevamento del pianto
        self.is_crying = False
        self.cry_detected_time = 0
    
    def initUI(self):
        """Inizializzazione dell'interfaccia utente"""
        self.setWindowTitle("Baby Cry Detector")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(STYLE)
        
        # Widget principale e layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Titolo
        title_label = QLabel("👶 Baby Cry Detector")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Descrizione
        desc_label = QLabel("Ascolta attraverso il microfono e riproduce audio rilassante quando viene rilevato il pianto di un bambino.")
        desc_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc_label)
        
        # Splitter principale per dividere configurazioni e stati
        splitter = QSplitter(Qt.Horizontal)
        
        # Pannello configurazioni
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # 1. Gruppo file audio
        self.create_audio_files_group(config_layout)
        
        # 2. Gruppo metodi di rilevamento
        self.create_detection_methods_group(config_layout)
        
        # 3. Gruppo microfono
        self.create_microphone_group(config_layout)
        
        # Aggiungi spaziatura elastica
        config_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Pannello stati
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        
        # 1. Gruppo stato in tempo reale
        self.create_realtime_status_group(status_layout)
        
        # 2. Visualizzazione audio
        self.create_audio_visualization_group(status_layout)
        
        # 3. Tab cronologia e file
        self.create_history_tabs(status_layout)
        
        # Aggiungi i widget al splitter
        splitter.addWidget(config_widget)
        splitter.addWidget(status_widget)
        splitter.setSizes([400, 800])  # Proporzioni iniziali
        
        main_layout.addWidget(splitter)
        
        # Pulsanti controllo
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶️ Avvia Rilevatore")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_detector)
        
        self.stop_button = QPushButton("⏹️ Ferma Rilevatore")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_detector)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(button_layout)
        
        # Istruzioni
        self.create_instructions(main_layout)
        
        self.setCentralWidget(main_widget)
        
        # Inizializzazioni finali
        self.update_mic_list()
    
    def create_audio_files_group(self, parent_layout):
        """Crea il gruppo per la gestione dei file audio"""
        audio_group = QGroupBox("File Audio")
        audio_layout = QVBoxLayout()
        
        # Pulsanti per aggiungere file
        add_button_layout = QHBoxLayout()
        self.add_files_button = QPushButton("➕ Aggiungi File")
        self.add_files_button.clicked.connect(self.add_audio_files)
        self.clear_files_button = QPushButton("🗑️ Cancella Tutto")
        self.clear_files_button.clicked.connect(self.clear_audio_files)
        
        add_button_layout.addWidget(self.add_files_button)
        add_button_layout.addWidget(self.clear_files_button)
        audio_layout.addLayout(add_button_layout)
        
        # Lista dei file
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        audio_layout.addWidget(self.file_list_widget)
        
        # Pulsante per rimuovere file selezionati
        self.remove_file_button = QPushButton("➖ Rimuovi Selezionati")
        self.remove_file_button.clicked.connect(self.remove_selected_files)
        audio_layout.addWidget(self.remove_file_button)
        
        # Modalità di riproduzione
        playback_label = QLabel("Modalità di riproduzione:")
        audio_layout.addWidget(playback_label)
        
        self.playback_mode_group = QButtonGroup()
        
        self.random_mode_radio = QRadioButton("Casuale")
        self.random_mode_radio.setChecked(True)
        self.sequential_mode_radio = QRadioButton("Sequenziale")
        self.all_mode_radio = QRadioButton("Tutti i file")
        
        self.playback_mode_group.addButton(self.random_mode_radio, 1)
        self.playback_mode_group.addButton(self.sequential_mode_radio, 2)
        self.playback_mode_group.addButton(self.all_mode_radio, 3)
        
        audio_layout.addWidget(self.random_mode_radio)
        audio_layout.addWidget(self.sequential_mode_radio)
        audio_layout.addWidget(self.all_mode_radio)
        
        audio_group.setLayout(audio_layout)
        parent_layout.addWidget(audio_group)
    
    def create_detection_methods_group(self, parent_layout):
        """Crea il gruppo per i metodi di rilevamento"""
        detection_group = QGroupBox("Metodo di Rilevamento")
        detection_layout = QVBoxLayout()
        
        # Radio button per i metodi
        self.detection_method_group = QButtonGroup()
        
        self.simple_method_radio = QRadioButton("Rilevamento livello semplice")
        self.simple_method_radio.setChecked(True)
        self.advanced_method_radio = QRadioButton("Rilevamento caratteristiche avanzate")
        self.ml_method_radio = QRadioButton("Rilevamento ML (modello pre-addestrato)")
        self.dl_method_radio = QRadioButton("Rilevamento DL (Hugging Face)")
        
        self.detection_method_group.addButton(self.simple_method_radio, 1)
        self.detection_method_group.addButton(self.advanced_method_radio, 2)
        self.detection_method_group.addButton(self.ml_method_radio, 3)
        self.detection_method_group.addButton(self.dl_method_radio, 4)
        
        # Collegamento per aggiornare sliders
        self.simple_method_radio.toggled.connect(self.toggle_threshold_sliders)
        self.advanced_method_radio.toggled.connect(self.toggle_threshold_sliders)
        self.ml_method_radio.toggled.connect(self.toggle_threshold_sliders)
        self.dl_method_radio.toggled.connect(self.toggle_threshold_sliders)
        
        detection_layout.addWidget(self.simple_method_radio)
        detection_layout.addWidget(self.advanced_method_radio)
        detection_layout.addWidget(self.ml_method_radio)
        detection_layout.addWidget(self.dl_method_radio)
        
        # Slider soglie
        slider_label = QLabel("Soglie di rilevamento:")
        detection_layout.addWidget(slider_label)
        
        # Simple threshold slider
        self.simple_threshold_label = QLabel("Soglia di livello semplice: 0.40")
        detection_layout.addWidget(self.simple_threshold_label)
        
        self.simple_threshold_slider = QSlider(Qt.Horizontal)
        self.simple_threshold_slider.setMinimum(10)
        self.simple_threshold_slider.setMaximum(100)
        self.simple_threshold_slider.setValue(40)
        self.simple_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.simple_threshold_slider.setTickInterval(10)
        self.simple_threshold_slider.valueChanged.connect(self.update_threshold_labels)
        detection_layout.addWidget(self.simple_threshold_slider)
        
        # Advanced/ML threshold slider
        self.advanced_threshold_label = QLabel("Soglia rilevamento avanzato: 0.70")
        detection_layout.addWidget(self.advanced_threshold_label)
        
        self.advanced_threshold_slider = QSlider(Qt.Horizontal)
        self.advanced_threshold_slider.setMinimum(10)
        self.advanced_threshold_slider.setMaximum(100)
        self.advanced_threshold_slider.setValue(70)
        self.advanced_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.advanced_threshold_slider.setTickInterval(10)
        self.advanced_threshold_slider.valueChanged.connect(self.update_threshold_labels)
        detection_layout.addWidget(self.advanced_threshold_slider)
        
        # Stato del modello ML
        self.ml_model_status = QLabel(f"Modello verrà caricato da: {DEFAULT_MODEL_PATH}")
        detection_layout.addWidget(self.ml_model_status)
        self.check_ml_model()
        
        # Imposta visibilità iniziale
        self.toggle_threshold_sliders()
        
        detection_group.setLayout(detection_layout)
        parent_layout.addWidget(detection_group)
    
    def create_microphone_group(self, parent_layout):
        """Crea il gruppo per la selezione del microfono"""
        mic_group = QGroupBox("Microfono")
        mic_layout = QVBoxLayout()
        
        mic_label = QLabel("Seleziona il microfono:")
        mic_layout.addWidget(mic_label)
        
        # Combo box per elenco microfoni
        self.mic_combo = QComboBox()
        mic_layout.addWidget(self.mic_combo)
        
        # Pulsante aggiorna microfoni
        self.refresh_mic_button = QPushButton("🔍 Aggiorna Lista Microfoni")
        self.refresh_mic_button.clicked.connect(self.update_mic_list)
        mic_layout.addWidget(self.refresh_mic_button)
        
        # Stato microfono
        self.mic_status_label = QLabel("Stato microfono: verifica in corso...")
        mic_layout.addWidget(self.mic_status_label)
        
        mic_group.setLayout(mic_layout)
        parent_layout.addWidget(mic_group)
    
    def create_realtime_status_group(self, parent_layout):
        """Crea il gruppo per lo stato in tempo reale"""
        status_group = QGroupBox("Stato in Tempo Reale")
        status_layout = QVBoxLayout()
        
        # Frame per lo stato
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")
        status_frame_layout = QVBoxLayout(status_frame)
        
        # Stato di esecuzione
        self.running_status_label = QLabel("🔴 INATTIVO")
        self.running_status_label.setObjectName("statusLabel")
        self.running_status_label.setAlignment(Qt.AlignCenter)
        status_frame_layout.addWidget(self.running_status_label)
        
        # Messaggio di stato
        self.status_message_label = QLabel("In attesa di avvio...")
        self.status_message_label.setAlignment(Qt.AlignCenter)
        status_frame_layout.addWidget(self.status_message_label)
        
        # Contatore rilevamenti
        self.detection_counter_label = QLabel("Pianti rilevati: 0")
        self.detection_counter_label.setAlignment(Qt.AlignCenter)
        status_frame_layout.addWidget(self.detection_counter_label)
        
        # Indicatore livello audio
        audio_level_label = QLabel("Livello Audio:")
        status_frame_layout.addWidget(audio_level_label)
        
        self.audio_level_progress = QProgressBar()
        self.audio_level_progress.setRange(0, 100)
        self.audio_level_progress.setValue(0)
        status_frame_layout.addWidget(self.audio_level_progress)
        
        # Stato riproduzione
        self.playing_status_label = QLabel("")
        self.playing_status_label.setAlignment(Qt.AlignCenter)
        status_frame_layout.addWidget(self.playing_status_label)
        
        # Metodo di rilevamento
        self.detection_method_label = QLabel("Metodo di rilevamento: Semplice")
        status_frame_layout.addWidget(self.detection_method_label)
        
        # Soglia attuale
        self.threshold_info_label = QLabel("Soglia: 0.40")
        status_frame_layout.addWidget(self.threshold_info_label)
        
        status_layout.addWidget(status_frame)
        status_group.setLayout(status_layout)
        parent_layout.addWidget(status_group)
    
    def create_audio_visualization_group(self, parent_layout):
        """Crea il gruppo per la visualizzazione dell'audio"""
        visualization_group = QGroupBox("Forma d'Onda Audio")
        visualization_layout = QVBoxLayout()
        
        # Canvas per il grafico audio
        self.audio_canvas = AudioPlotCanvas(self, width=5, height=3, dpi=100)
        visualization_layout.addWidget(self.audio_canvas)
        
        visualization_group.setLayout(visualization_layout)
        parent_layout.addWidget(visualization_group)
    
    def create_history_tabs(self, parent_layout):
        """Crea i tab per cronologia e file"""
        tab_widget = QTabWidget()
        
        # Tab cronologia
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setText("Rilevamenti recenti: Nessuno")
        history_layout.addWidget(self.history_text)
        
        # Tab file
        files_tab = QWidget()
        files_layout = QVBoxLayout(files_tab)
        
        self.files_text = QTextEdit()
        self.files_text.setReadOnly(True)
        self.files_text.setText("File audio (0): Nessuno")
        files_layout.addWidget(self.files_text)
        
        # Aggiungi i tab
        tab_widget.addTab(history_tab, "Cronologia Rilevamenti")
        tab_widget.addTab(files_tab, "Dettagli File Audio")
        
        parent_layout.addWidget(tab_widget)
    
    def create_instructions(self, parent_layout):
        """Crea le istruzioni"""
        instructions_group = QGroupBox("Istruzioni")
        instructions_layout = QVBoxLayout()
        
        instructions_text = QTextEdit()
        instructions_text.setReadOnly(True)
        instructions_text.setHtml("""
        <ol>
            <li>Carica uno o più file audio (ninne nanne, voce della madre, rumore bianco, ecc.)</li>
            <li>Scegli un metodo di rilevamento:
                <ul>
                    <li><b>Rilevamento livello semplice</b>: Si attiva quando il livello audio supera la soglia</li>
                    <li><b>Rilevamento caratteristiche avanzate</b>: Usa caratteristiche audio per un rilevamento più accurato</li>
                    <li><b>Rilevamento ML</b>: Usa un modello di machine learning pre-addestrato per la massima precisione</li>
                    <li><b>Rilevamento DL</b>: Usa un modello di deep learning da Hugging Face per la massima accuratezza</li>
                </ul>
            </li>
            <li>Scegli una modalità di riproduzione:
                <ul>
                    <li><b>Casuale</b>: Riproduce un file casuale ogni volta</li>
                    <li><b>Sequenziale</b>: Cicla attraverso i file in ordine</li>
                    <li><b>Tutti i file</b>: Riproduce tutti i file uno dopo l'altro</li>
                </ul>
            </li>
            <li>Seleziona il microfono dal menu a tendina</li>
            <li>Clicca su "Avvia Rilevatore" per iniziare l'ascolto</li>
            <li>Lo stato e la forma d'onda audio si aggiornano in tempo reale</li>
            <li>Clicca su "Ferma Rilevatore" quando hai finito</li>
        </ol>
        """)
        instructions_layout.addWidget(instructions_text)
        
        instructions_group.setLayout(instructions_layout)
        parent_layout.addWidget(instructions_group)
    
    def toggle_threshold_sliders(self):
        """Aggiorna la visibilità degli slider in base al metodo selezionato"""
        if self.simple_method_radio.isChecked():
            self.simple_threshold_label.setVisible(True)
            self.simple_threshold_slider.setVisible(True)
            self.advanced_threshold_label.setVisible(False)
            self.advanced_threshold_slider.setVisible(False)
            self.ml_model_status.setVisible(False)
        else:
            self.simple_threshold_label.setVisible(False)
            self.simple_threshold_slider.setVisible(False)
            self.advanced_threshold_label.setVisible(True)
            self.advanced_threshold_slider.setVisible(True)
            
            # Mostra lo stato del modello solo se è selezionato ML
            self.ml_model_status.setVisible(self.ml_method_radio.isChecked())
    
    def update_threshold_labels(self):
        """Aggiorna le etichette delle soglie quando cambiano i valori degli slider"""
        simple_value = self.simple_threshold_slider.value() / 100
        advanced_value = self.advanced_threshold_slider.value() / 100
        
        self.simple_threshold_label.setText(f"Soglia di livello semplice: {simple_value:.2f}")
        self.advanced_threshold_label.setText(f"Soglia rilevamento avanzato: {advanced_value:.2f}")
    
    def update_mic_list(self):
        """Aggiorna l'elenco dei microfoni disponibili"""
        self.mic_combo.clear()
        mic_choices = self.detector.get_available_mics()
        
        for id, name in mic_choices:
            self.mic_combo.addItem(name, id)
        
        # Aggiorna lo stato del microfono
        if self.detector.mic_error:
            self.mic_status_label.setText(f"⚠️ Errore microfono: {self.detector.mic_error}")
        else:
            self.mic_status_label.setText(f"✅ Microfono: {self.detector.current_mic_name}")
    
    def add_audio_files(self):
        """Apre un dialogo per selezionare i file audio da aggiungere"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleziona file audio", "", 
            "File Audio (*.mp3 *.wav *.ogg *.flac *.m4a);;Tutti i file (*)"
        )
        
        if files:
            # Processa i file selezionati
            for file in files:
                if file not in self.detector.voice_files:
                    self.detector.voice_files.append(file)
                    self.file_list_widget.addItem(os.path.basename(file))
            
            # Aggiorna il testo dei file
            self.update_files_text()
    
    def remove_selected_files(self):
        """Rimuove i file selezionati dalla lista"""
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            row = self.file_list_widget.row(item)
            self.file_list_widget.takeItem(row)
            filename = item.text()
            
            # Trova e rimuovi il percorso completo dai file del rilevatore
            for i, file_path in enumerate(self.detector.voice_files):
                if os.path.basename(file_path) == filename:
                    self.detector.voice_files.pop(i)
                    break
        
        # Aggiorna il testo dei file
        self.update_files_text()
    
    def clear_audio_files(self):
        """Cancella tutti i file audio"""
        self.file_list_widget.clear()
        self.detector.voice_files = []
        self.update_files_text()
    
    def update_files_text(self):
        """Aggiorna il testo informativo dei file"""
        file_text = format_file_list(
            self.detector.voice_files, 
            self.detector.current_file_index, 
            self.detector.playback_mode
        )
        self.files_text.setText(file_text)
    
    def get_selected_detection_method(self):
        """Ottiene il metodo di rilevamento selezionato"""
        if self.simple_method_radio.isChecked():
            return "simple"
        elif self.advanced_method_radio.isChecked():
            return "advanced"
        elif self.ml_method_radio.isChecked():
            return "ml"
        elif self.dl_method_radio.isChecked():
            return "dl"
        return "simple"  # Default
    
    def get_selected_playback_mode(self):
        """Ottiene la modalità di riproduzione selezionata"""
        if self.random_mode_radio.isChecked():
            return "random"
        elif self.sequential_mode_radio.isChecked():
            return "sequential"
        elif self.all_mode_radio.isChecked():
            return "all"
        return "random"  # Default
    
    def check_ml_model(self):
        """Verifica l'esistenza del modello ML"""
        if os.path.exists(DEFAULT_MODEL_PATH):
            self.ml_model_status.setText(f"✅ Modello ML trovato: {DEFAULT_MODEL_PATH}")
        else:
            self.ml_model_status.setText(
                f"ℹ️ Modello ML non trovato: {DEFAULT_MODEL_PATH}. "
                "Verrà creato automaticamente un modello fittizio."
            )
    
    def start_detector(self):
        """Avvia il rilevatore con i parametri selezionati"""
        # Se il rilevatore è già in esecuzione, fermalo e resetta tutto prima di riavviare
        if self.detector.is_running:
            self.stop_detector()
        
        # Ottieni parametri dall'interfaccia
        detection_method = self.get_selected_detection_method()
        playback_mode = self.get_selected_playback_mode()
        simple_threshold = self.simple_threshold_slider.value() / 100
        advanced_threshold = self.advanced_threshold_slider.value() / 100
        selected_mic = self.mic_combo.currentData()
        
        # Controlla se ci sono file audio
        if not self.detector.voice_files:
            # Aggiungi un file predefinito se non ce ne sono
            default_file = "voce_madre.mp3"
            if os.path.exists(default_file):
                self.detector.voice_files = [default_file]
                self.file_list_widget.clear()
                self.file_list_widget.addItem(default_file)
                self.update_files_text()
        
        # Determina il percorso del modello se è selezionato ML
        model_path = None
        if detection_method == "ml":
            model_path = DEFAULT_MODEL_PATH
        
        # Avvia il rilevatore
        self.detector.start_detection(
            voice_files=self.detector.voice_files,
            detection_method=detection_method,
            model_path=model_path,
            threshold=advanced_threshold,
            simple_threshold=simple_threshold,
            playback_mode=playback_mode,
            mic_id=selected_mic
        )
        
        # Avvia il thread di aggiornamento
        if self.detector_thread is None or not self.detector_thread.isRunning():
            self.detector_thread = DetectorThread(self.detector)
            self.detector_thread.status_update.connect(self.process_status_update)
            self.detector_thread.start()
        
        # Aggiorna l'interfaccia
        self.update_ui()
    
    def stop_detector(self):
        """Ferma il rilevatore e resetta completamente lo stato"""
        # Ferma il rilevatore attuale
        self.detector.stop_detection()
        
        # Ferma il thread di aggiornamento
        if self.detector_thread and self.detector_thread.isRunning():
            self.detector_thread.stop()
            self.detector_thread.wait()
            self.detector_thread = None
        
        # RESET COMPLETO: Crea una nuova istanza del rilevatore
        # Salvare prima alcuni parametri che vogliamo mantenere
        voice_files = self.detector.voice_files
        selected_mic_id = self.detector.selected_mic_id
        
        # Crea una nuova istanza pulita
        self.detector = BabyCryDetector()
        
        # Ripristina i file audio e il microfono selezionato
        self.detector.voice_files = voice_files
        self.detector.selected_mic_id = selected_mic_id
        self.detector._test_microphone()  # Verifica microfono
        
        # Aggiorna l'interfaccia con i valori resettati
        self.running_status_label.setText("🔴 INATTIVO")
        self.status_message_label.setText("In attesa di avvio...")
        self.detection_counter_label.setText("Pianti rilevati: 0")
        self.playing_status_label.setText("")
        self.history_text.setText("Rilevamenti recenti: Nessuno")
        
        # Reset audio progress bar
        self.audio_level_progress.setValue(0)
        
        # Reset grafico audio
        self.audio_canvas.update_plot(
            np.zeros(1000),  # Reset dati audio
            0.0,             # Reset livello audio
            mic_name=self.detector.current_mic_name,
            num_files=len(self.detector.voice_files),
            playback_mode=self.detector.playback_mode,
            is_crying=False,
            detection_method="Rilevatore inattivo"
        )
        
        # Aggiorna il testo dei file
        self.update_files_text()
        
        # Aggiorna l'UI
        self.update_ui()
    
    def process_status_update(self, status):
        """Elabora gli aggiornamenti di stato dal thread del rilevatore"""
        # Aggiorna l'interfaccia con i nuovi valori
        self.running_status_label.setText(status["running_status"])
        self.status_message_label.setText(status["status_message"])
        self.detection_counter_label.setText(status["detection_count"])
        self.playing_status_label.setText(status["playing_status"])
        self.history_text.setText(status["history_text"])
        self.files_text.setText(status["file_text"])
        self.detection_method_label.setText(status["detection_method"])
        self.threshold_info_label.setText(status["threshold_info"])
        self.mic_status_label.setText(status["mic_status"])
        
        # Aggiorna la barra di progresso del livello audio
        level = int(self.detector.current_audio_level * 100)
        self.audio_level_progress.setValue(level)
        
        # Cambia il colore in base al livello
        if level < 30:
            self.audio_level_progress.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        elif level < 70:
            self.audio_level_progress.setStyleSheet("QProgressBar::chunk { background-color: #FFC107; }")
        else:
            self.audio_level_progress.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }")
        
        # Verifica se è stato rilevato un pianto
        current_time = time.time()
        if current_time - self.detector.last_detected < 3:
            self.is_crying = True
            self.cry_detected_time = self.detector.last_detected
        else:
            self.is_crying = False
        
        # Aggiorna il grafico audio
        threshold = self.simple_threshold_slider.value() / 100 if self.simple_method_radio.isChecked() else None
        self.audio_canvas.update_plot(
            self.detector.audio_data_for_plot,
            self.detector.current_audio_level,
            threshold=threshold,
            mic_name=self.detector.current_mic_name,
            num_files=len(self.detector.voice_files),
            playback_mode=self.detector.playback_mode,
            is_crying=self.is_crying,
            detection_method=self.detector.get_detector_name()
        )
    
    def update_ui(self):
        """Aggiorna l'interfaccia periodicamente"""
        if not self.detector.is_running and not self.detector_thread:
            # Aggiorna lo stato del microfono
            if self.detector.mic_error:
                self.mic_status_label.setText(f"⚠️ Errore microfono: {self.detector.mic_error}")
            else:
                self.mic_status_label.setText(f"✅ Microfono: {self.detector.current_mic_name}")
            
            # Controlla lo stato del modello ML
            self.check_ml_model()
            
            # Aggiorna la lista dei file
            self.file_list_widget.clear()
            for file in self.detector.voice_files:
                self.file_list_widget.addItem(os.path.basename(file))
            
            # Aggiorna il testo dei file
            self.update_files_text()

def main():
    """Funzione principale per l'avvio dell'applicazione"""
    # Verifica esistenza directory models
    os.makedirs("models", exist_ok=True)
    
    app = QApplication(sys.argv)
    window = BabyCryDetectorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()