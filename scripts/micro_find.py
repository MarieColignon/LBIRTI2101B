import os
import torch
import wespeaker

# --- Charger le modèle ---
model = wespeaker.load_model('english')
model.set_device('cuda:0')

# --- Dossier des embeddings ---
racine_embeddings = "C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/embeddings"

# --- Fonction de reconnaissance ---
def recognize(audio_path: str):
    q = model.extract_embedding(audio_path)  # embedding du fichier audio à tester
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

    return {
        'name': best_name,
        'confidence': best_score
    }

# --- Test ---
audio_test = "C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/enregistrement/vincentf/VincentF_1.wav"
result = recognize(audio_test)
print(f"Identifié comme : {result['name']} (confiance : {result['confidence']:.2f})")
