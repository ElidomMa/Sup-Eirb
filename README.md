# Sup-Eirb

Comparaison de 4 modèles de super-résolution : EDSR, SPAN, Real-ESRGAN et notre U-Net.

## Avant de lancer

Ajouter les images de test dans `Images/test/` (dossiers `HR`, `LR_X2`, `LR_X3`, `LR_X4`).

Ajouter les poids des modèles :

- EDSR : `Existing_Models/EDSR/models/`
- SPAN : `Existing_Models/SPAN/models/`
- U-Net : `Existing_Models/UNet/checkpoints/x2/`, `x3/` et `x4/`
- Real-ESRGAN : `Existing_Models/Real-ERSGAN/weights/`

## Lancer les tests

```bash
./lancer_tests.sh
```

Les résultats sont enregistrés dans le dossier de chaque modèle, et les images agrandies dans son dossier `results/`.
