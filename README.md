# 👶 Baby Cry Detector 🎶

This project implements a real-time baby cry detection system using Pretrained Audio Neural Networks (PANNs). When a baby's cry is detected through the microphone, the application can play a pre-recorded mother's voice to soothe the baby. The interface is built using Gradio.

## ✨ Features

*   🎤 **Real-time Audio Monitoring**: Listens to microphone input continuously.
*   🧠 **AI-Powered Detection**: Uses a PANNs model (Cnn14) to identify baby cry sounds.
*   🗣️ **Mother's Voice Playback**: Plays a randomly selected mother's voice recording upon cry detection.
*   🖥️ **Web Interface**: Easy-to-use Gradio UI to upload voice files, start/stop monitoring, and view logs.
*   📊 **Detection Scores**: Displays the confidence score for baby cry detection.
*   ⚙️ **Configurable**: Adjustable parameters like detection threshold (though currently hardcoded, can be exposed).
*   📝 **Logging**: Provides real-time logs of detection events and system status.
*   🛑 **Graceful Stop**: Allows stopping the monitoring process cleanly.

## 📋 Requirements

### Software:
*   Python (3.8+ recommended)
*   `pip` (Python package installer)

### Python Libraries:
You can install these using the `requirements.txt` file provided below.
*   `gradio`
*   `numpy`
*   `pyaudio`
*   `panns_inference` (The PANNs model inference library)
*   `pydub`
*   `torch` (PyTorch - CPU version is fine, GPU/MPS will offer better performance if available)

### System Dependencies:
*   **For PyAudio**:
    *   **Windows**: Usually works out of the box with pip install.
    *   **Linux**: You might need to install `portaudio` development libraries (e.g., `sudo apt-get install portaudio19-dev python3-pyaudio`).
    *   **macOS**: You might need to install `portaudio` (e.g., `brew install portaudio`).

### Model Weights:
*   The PANNs model checkpoint file: `Cnn14_DecisionLevelMax_mAP=0.385.pth`.
    *   This file should be placed in a `weights` directory within your project's root folder (i.e., `CryBabyParentDetector/weights/Cnn14_DecisionLevelMax_mAP=0.385.pth`).
    *   You'll need to acquire this weights file. It's commonly available from PANNs-related repositories or resources.

## 🛠️ Setup & Installation

1.  **Clone the Repository (if applicable) or Download Files:**
    Ensure you have all the project files, including `pann_real_time_interface.py`.

2.  **Create a `weights` Directory:**
    Inside your project folder (`CryBabyParentDetector`), create a directory named `weights`.
    ```bash
    mkdir weights
    ```

3.  **Place Model Weights:**
    Download the `Cnn14_DecisionLevelMax_mAP=0.385.pth` file and place it inside the `CryBabyParentDetector/weights/` directory.

4.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If you encounter issues with `PyAudio`, please refer to its installation guide for your specific OS and ensure system dependencies (like `portaudio`) are met.

## 🚀 How to Run

1.  **Navigate to the Project Directory:**
    Open your terminal or command prompt and go to the `CryBabyParentDetector` directory.
    ```bash
    cd path\to\CryBabyParentDetector
    ```

2.  **Run the Application:**
    ```bash
    python pann_real_time_interface.py
    ```

3.  **Open the Web Interface:**
    The script will output a URL (usually `http://127.0.0.1:7860` or similar, and possibly a public Gradio link if `share=True` is used). Open this URL in your web browser.

4.  **Using the Interface:**
    *   **Upload Mom's Voice Files**: Click the upload area to select one or more MP3 files of the mother's voice.
    *   **Start Monitoring**: Click the "Start Monitoring" button. The application will start listening to your microphone.
    *   **View Logs**: Detection events, scores, and status messages will appear in the "Detection Log" area.
    *   **Stop Monitoring**: Click the "Stop Monitoring" button to halt the detection process.

## 💡 Notes & Troubleshooting

*   **Microphone Access**: Ensure the application has permission to access your microphone.
*   **Model Loading**: If you see "Error - Model NOT loaded" or "label NOT found" messages:
    *   Verify the `weights/Cnn14_DecisionLevelMax_mAP=0.385.pth` file is correctly placed.
    *   Check console output for more detailed error messages related to PyTorch or model file issues.
    *   The `BABY_IDX` error means the label "Baby cry, infant cry" wasn't found in the model's label set, which could indicate an issue with the `panns_inference` library or its version.
*   **Performance**: Detection performance (speed and accuracy) can depend on your CPU/GPU. If using a GPU, ensure PyTorch is correctly configured to use it.
*   **PyAudio Errors**: Common issues include incorrect microphone selection or driver problems. The script attempts to use the default input device.
*   **Input Overflow**: If you see "PyAudio input overflowed" warnings, the system might not be processing audio fast enough. This could be due to high CPU load or a `HOP_SIZE` that's too small for the processing time.
