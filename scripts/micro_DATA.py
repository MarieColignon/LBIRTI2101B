import os
import torch
import torchaudio
import sounddevice as sd


# === PARAMÈTRES ===
frequence = 16000  # compatible WeSpeaker
racine = "C:/Users/Vincent/OneDrive/Documents/GitHub/LBIRTI2101B/DATA/enregistrement"
os.makedirs(racine, exist_ok=True)

print("Système d’enregistrement vocal multi-personnes")
print("Tape 'q' à tout moment pour quitter.\n")

while True:
    # --- Choisir la personne ---
    nom_personne = input("Nom de la personne (ou 'q' pour quitter) : ").strip()
    if nom_personne.lower() == "q":
        print("\n Fin de session.")
        break

    # Créer le dossier de cette personne
    dossier_personne = os.path.join(racine, nom_personne.lower().replace(" ", "_"))
    os.makedirs(dossier_personne, exist_ok=True)

    # Compter combien d’enregistrements existent déjà
    existants = [f for f in os.listdir(dossier_personne) if f.endswith(".wav")]
    index = len(existants) + 1

    print(f"\n Enregistrements pour {nom_personne} (dossier : {dossier_personne})")
    print("Appuie sur Entrée pour lancer un enregistrement, ou 'n' pour passer à une autre personne.\n")

    while True:
        choix = input("▶️  Démarrer un nouvel enregistrement ? (Entrée = Oui / n = suivant / q = quitter) : ").strip().lower()
        if choix == "q":
            print("\n Fin de session.")
            exit()
        if choix == "n":
            print(f" Passage à une autre personne.\n")
            break

        # Nom du fichier audio
        fichier_audio = os.path.join(dossier_personne, f"{nom_personne}_{index}.wav")

        # --- Enregistrement audio ---
        input(f"Appuie sur Entrée pour démarrer l’enregistrement de {nom_personne}_{index}...")
        print(" Enregistrement en cours... (Ctrl+C pour arrêter)")

        frames = []
        try:
            with sd.InputStream(samplerate=frequence, channels=1, dtype='float32') as stream:
                while True:
                    data, _ = stream.read(1024)
                    frames.append(torch.from_numpy(data.copy()))
        except KeyboardInterrupt:
            print(" Enregistrement terminé.\n")

        # --- Sauvegarde du fichier .wav ---
        audio_tensor = torch.cat(frames, dim=0).T
        torchaudio.save(fichier_audio, audio_tensor, frequence)
        print(f" Sauvegardé : {fichier_audio}\n")

        index += 1
