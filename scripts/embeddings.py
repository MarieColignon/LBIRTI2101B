import torch
import os
import wespeaker

#chargement du modèle pré-entraîné
model = wespeaker.load_model('english')
model.set_device('cuda:0')


# Chemins des dossiers
racine_audio = "C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/enregistrement/"
racine_embeddings = "C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/embeddings"
os.makedirs(racine_embeddings, exist_ok=True)



# 3. Parcourir toutes les personnes
for personne in os.listdir(racine_audio):
    dossier_personne = os.path.join(racine_audio, personne)
    if not os.path.isdir(dossier_personne):
        continue

    print(f"Traitement de {personne} ...")
    embeddings = []

    for fichier in os.listdir(dossier_personne):
        if fichier.endswith(".wav"):
            chemin_audio = os.path.join(dossier_personne, fichier)
            emb = model.extract_embedding(chemin_audio)
            embeddings.append(emb)
    
    if embeddings:
        # 5. Moyenne des embeddings
        moyenne = torch.stack(embeddings).mean(dim=0)


        # 6. Sauvegarder le vecteur moyen
        chemin_embed = os.path.join(racine_embeddings, f"{personne}.pt")
        torch.save(moyenne, chemin_embed)
        print(f"Embedding moyen sauvegardé pour {personne} : {chemin_embed}")
