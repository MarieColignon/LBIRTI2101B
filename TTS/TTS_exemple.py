
# 2️⃣ Chemin vers le fichier audio de référence (pour cloner la voix)
racine_audio = "C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/enregistrement/vincentf/vincentf_7.wav"
from TTS.api import TTS
import soundfile as sf
import numpy as np

tts = TTS("tts_models/multilingual/multi-dataset/your_tts")


text = "Bonjour, ceci est une voix clonée. Je teste la synthèse vocale avec clonage de voix."

speaker_name = "vincentf"
# 4) Synthèse
wav = tts.tts(
    text=text,
    speaker=speaker_name,
    language='fr-fr'
)

sf.write("output.wav", wav, tts.synthesizer.output_sample_rate)
print("OK")
