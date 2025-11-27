

import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


AUDIO_PATH = Path('C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/data_clean')


# Récupérer tous les fichiers WAV (avec l'extension)
audio_files = list(AUDIO_PATH.glob("*.wav"))

print("Fichiers audio trouvés :", audio_files)

# Charger le premier fichier
if audio_files:
    waveform, sr = torchaudio.load(audio_files[0], format="wav", backend="soundfile")
    print("Loaded:", audio_files[0])
else:
    print("Aucun fichier WAV trouvé dans le dossier !")


# AUDIO_PATH = "./audio.wav"
print('2')
# ---------------- Load ----------------
   # waveform: (channels, samples)
print("Loaded:", AUDIO_PATH)
print("waveform.shape:", waveform.shape, "sample_rate:", sr)

# Mix to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)  # shape -> (1, N)

# Convert to 1D signal for some ops
signal = waveform[0]   # shape (N,)
N = signal.numel()

# ---------------- Waveform plot ----------------
plt.figure(figsize=(10, 2.5))
times = torch.arange(N) / sr
plt.plot(times.numpy(), signal.numpy())
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.xlim(0, times[-1].item())
plt.tight_layout()
plt.show()

# ---------------- FFT (rfft) ----------------
# Use real FFT (rfft) -> positive frequencies only
fft_complex = torch.fft.rfft(signal)          # complex tensor length N//2 + 1
magnitude = torch.abs(fft_complex) / N        # normalize by N (optional)
freqs = torch.fft.rfftfreq(n=N, d=1.0/sr)     # frequencies corresponding to bins

plt.figure(figsize=(10, 3))
plt.plot(freqs.numpy(), magnitude.numpy())
plt.xlim(0, sr/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (norm.)")
plt.title("FFT (rfft) — Magnitude spectrum (positive freqs)")
plt.tight_layout()
plt.show()

# ---------------- STFT (torch.stft) ----------------
# Choose common TTS params (25 ms / 10 ms)
win_ms = 25
hop_ms = 10
n_fft = int(round(sr * win_ms / 1000.0))
hop_length = int(round(sr * hop_ms / 1000.0))
win_length = n_fft
window = torch.hann_window(win_length)

# torch.stft returning complex (return_complex=True)
stft_complex = torch.stft(signal, n_fft=n_fft, hop_length=hop_length,
                          win_length=win_length, window=window, return_complex=True)
S = torch.abs(stft_complex)   # shape: (freq_bins, time_frames)
S_db = 20 * torch.log10(S + 1e-10)  # amplitude to dB (approx)

plt.figure(figsize=(10, 4))
# display with imshow: frequency axis (0..n_fft//2) -> convert to Hz for ticks if desired
plt.imshow(S_db.numpy(), origin='lower', aspect='auto',
           extent=[0, (S.shape[1]*hop_length)/sr, 0, sr/2])
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title(f"STFT spectrogram — n_fft={n_fft}, hop={hop_length}")
plt.colorbar(label="dB (approx)")
plt.tight_layout()
plt.show()

# ---------------- Mel-spectrogram (torchaudio) ----------------
n_mels = 80  # feat_dim required
mel_transform = T.MelSpectrogram(
    sample_rate=sr,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    n_mels=n_mels,
    power=2.0,   # power spectrogram
)
mel_spec = mel_transform(waveform)   # shape: (channel, n_mels, time)
# convert to dB
to_db = T.AmplitudeToDB(stype='power')  # because power=2.0
mel_db = to_db(mel_spec)

# Sauvegarde Mel-spectrogramme en numpy
np.save('C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/mel_spect',mel_db[0].numpy())


# Plot mel (channel 0)
plt.figure(figsize=(10, 4))
m = mel_db[0].cpu().numpy()
plt.imshow(m, origin='lower', aspect='auto',
           extent=[0, (m.shape[1]*hop_length)/sr, 0, n_mels])
plt.xlabel("Time (s)")
plt.ylabel("Mel bin")
plt.title(f"Mel-spectrogram — n_mels={n_mels}")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()

print("STFT shape (freq_bins, time_frames):", S.shape)
print("Mel shape (channel, n_mels, time_frames):", mel_spec.shape)