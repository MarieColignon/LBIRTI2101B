import os
import torch
import sounddevice as sd
import wespeaker

# --- Paramètres ---
frequence = 16000  # compatible WeSpeaker
duree = 2         # secondes par segment
racine_embeddings = "C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/embeddings"

# --- Charger le modèle ---
model = wespeaker.load_model('english')
model.set_device('cuda:0')

# --- Fonction de reconnaissance en direct depuis un tensor ---
def recognize_tensor(audio_tensor: torch.Tensor, sample_rate: int = 16000):
    q = model.extract_embedding_from_pcm(audio_tensor, sample_rate)
    best_score = 0.0
    best_name = ''

    # Parcourir tous les fichiers .pt dans le dossier
    for fichier in os.listdir(racine_embeddings):
        if not fichier.endswith(".pt"):
            continue
        nom_personne = fichier.replace(".pt", "")
        emb_path = os.path.join(racine_embeddings, fichier)
        e = torch.load(emb_path)
        score = model.cosine_similarity(q, e)
        if score > best_score:
            best_score = score
            best_name = nom_personne

    return best_name, best_score

# --- Boucle principale ---
print(f"🎤 Reconnaissance vocale en direct, segment de {duree}s. Ctrl+C pour quitter.\n")

try:
    while True:
        print("🔴 Enregistrement en cours... Parle maintenant !")
        recording = sd.rec(int(duree * frequence), samplerate=frequence, channels=1, dtype='float32')
        sd.wait()

        audio_tensor = torch.from_numpy(recording.T)
        nom, score = recognize_tensor(audio_tensor)

        print(f"🟢 Identifié comme : {nom} (confiance : {score:.2f})\n")

except KeyboardInterrupt:
    print("\nFin de la reconnaissance en direct.")
