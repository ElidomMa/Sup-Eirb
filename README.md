# Sup-Eirb

Comparaison de 4 modèles de super-résolution : EDSR, SPAN, Real-ESRGAN et notre U-Net.

## Installation

Utiliser Python 3.10, avec PyTorch 2.1 et torchvision 0.16 (avec des versions plus récentes, le test de Real-ESRGAN ne fonctionne pas) :

```bash
pip install torch==2.1.2 torchvision==0.16.2
```

Puis installer les bibliothèques de chaque modèle :

```bash
pip install -r Existing_Models/EDSR/requirements.txt
pip install -r Existing_Models/SPAN/requirements.txt
pip install -r Existing_Models/Real-ERSGAN/requirements.txt
pip install numpy pillow
```

## Avant de lancer

Ajouter les images de test dans `Images/test/` (dossiers `HR`, `LR_X2`, `LR_X3`, `LR_X4`).

Ajouter les poids de chaque modèle :

- EDSR : `edsrx2.pth`, `edsrx3.pth` et `edsrx4.pth` dans `Existing_Models/EDSR/models/`
- SPAN : `spanx2_ch52.pth`, `spanx3_ch52.pth` et `spanx4_ch52.pth` dans `Existing_Models/SPAN/models/`
- Real-ESRGAN : `RealESRGAN_x2plus.pth` et `RealESRGAN_x4plus.pth` dans `Existing_Models/Real-ERSGAN/weights/`
- U-Net : `unet_x2_best.pt` dans `Existing_Models/UNet/checkpoints/x2/` (de même pour x3 et x4)

## Lancer les tests

```bash
./lancer_tests.sh
```

Les résultats sont enregistrés dans le dossier de chaque modèle, et les images agrandies dans son dossier `results/`.