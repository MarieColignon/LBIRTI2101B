import torchaudio
import matplotlib.pyplot as plt

# Charger l'audio
waveform, sr = torchaudio.load("harvard.wav")

# Si stéréo, prendre seulement le canal 0 (gauche) ou faire la moyenne
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)  # moyenne des canaux

# Générer le Mel spectrogramme
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=400,
    hop_length=160,
    n_mels=80
)
mel_spec = mel_transform(waveform)

# Convertir en décibels
mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

# Plot simple
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec_db.squeeze().numpy(), origin="lower", aspect="auto")
plt.title("Spectrogramme de Mel (torchaudio)")
plt.xlabel("Temps (frames)")
plt.ylabel("Bandes Mel")
plt.colorbar(label="dB")
plt.show()

# coucou une modif