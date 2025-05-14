import torchaudio
import torch

input_path = "/Users/cesaredavidepace/Desktop/Progetti/BabyCryDetector/strange.mp3"
# 1) Load MP3 (any sample rate, stereo or mono)
waveform, orig_sr = torchaudio.load(input_path)  # → Tensor shape (channels, samples)

# 2) Resample to 16 kHz & mix down to mono
if orig_sr != 32000:
    waveform = torchaudio.functional.resample(waveform, orig_sr, 32000)
# mix to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# 3) Convert to numpy for HF pipeline (which will still check sample rate)
audio_np = waveform.squeeze(0).numpy()

print(orig_sr)
output_path = input_path.replace(".mp3", ".wav") #TODO

# 4) Save as WAV
torchaudio.save(output_path, waveform, 32000)