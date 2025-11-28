
import os
import librosa
import torch
import torchaudio
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

checkpoint_dir = r"C:/Users/Vincent/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2"   # <-- Dossier contenant: model.pth, config.json, vocab.json, speakers_xtts.pth

print("Loading model...")

config = XttsConfig()
config.load_json(f"{checkpoint_dir}/config.json")

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=checkpoint_dir,
    use_deepspeed=False
)

model.cuda()


# ------------------------------
#       EXTRACTION DES LATENTS
# ------------------------------

reference_audio = "C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/enregistrement/vincentf/VincentF_1.wav"

print("Computing speaker latents...")

gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[reference_audio]
)

torch.save(gpt_cond_latent, "gpt_latent.pt")
torch.save(speaker_embedding, "speaker_embedding.pt")

print("Latents extracted !")
print("gpt_cond_latent shape:", gpt_cond_latent.shape)
print("speaker_embedding shape:", speaker_embedding.shape)