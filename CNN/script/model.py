import wespeaker
from wespeaker.models.resnet import ResNet34

# Créer le modèle pré-entraîné
model = ResNet34(feat_dim=80, embed_dim=256)

print(model)
