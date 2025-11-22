import TTS


from TTS.api import TTS
import numpy as np

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Bonjour à tous !",
    speaker_wav=None,
    speaker_embedding="C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/embeddings/vincentf.pty",
    language="fr",
    file_path="out.wav"
)

print("✅ Audio généré : out.wav")

