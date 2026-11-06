# Sup-Eirb

Sup-Eirb est une démonstration simple de super-résolution en temps réel. Au
démarrage, l'utilisateur choisit deux modèles réellement disponibles et un
facteur compatible. La même image basse résolution est envoyée aux deux
workers, puis les résultats sont affichés côte à côte.

```text
jeu choisi -> capture QtWebEngine ou MSS -> réduction 320/480/640
                                      +-> Bicubique x2 -> panneau gauche
                                      +-> ESPCN x2 CUDA -> panneau droit
```

## Fichiers

- `main.py` : lance l'application.
- `game.py` : navigateur intégré, clavier, souris, Pointer Lock et capture.
- `capture.py` : liste des fenêtres X11 et capture MSS de Firefox.
- `super_resolution.py` : modèles, facteurs réellement supportés et détection.
- `sequence.py` : tri numérique, lecture bornée et historique des séquences.
- `unet_adapter.py` : adaptateur séparé pour le U-Net original.
- `interface.py` : fenêtre redimensionnable, séparateur, mesures et plein écran.
- `models/ESPCN_x2.onnx` : modèle pré-entraîné ESPCN.

## Installation et lancement

L'environnement `.venv` étant déjà fourni :

```bash
cd ~/Sup-Eirb
python -m pip install -r requirements.txt
python main.py
```

La fenêtre de configuration propose deux modèles et un facteur commun. La barre
supérieure propose ensuite quatre sources :

1. saisir une adresse puis cliquer sur **Ouvrir** pour jouer dans QtWebEngine;
2. cliquer sur **Capturer un jeu Firefox**, choisir une fenêtre puis toute la
   fenêtre ou une région dessinée à la souris.
3. cliquer sur **Ouvrir une image**;
4. cliquer sur **Ouvrir une séquence** pour lire un dossier PNG/JPEG avec
   lecture, pause, précédent, suivant et recommencer.

Dans le second mode, les images traitées sont une copie MSS : le jeu reste
contrôlé depuis Firefox et non depuis les panneaux de comparaison. Ce mode
nécessite Ubuntu X11. Un jeu Firefox en plein écran peut recouvrir ou masquer
la source sur un écran unique; le navigateur intégré est alors recommandé.

La résolution d'entrée est indépendante de la taille de la fenêtre :
`320 × 180` privilégie la vitesse, `480 × 270` est équilibrée et `640 × 360`
privilégie la qualité. L'image est ajustée sans déformation, puis la même
image basse résolution est envoyée aux deux méthodes. Les images agrandies
dans les panneaux ne relancent aucun traitement.

La capture propose `5`, `10`, `15`, `30` et `60 FPS`. Ce réglage
correspond à la fréquence demandée au timer, pas à une garantie : le FPS réel
de capture est mesuré dans l'application. QtWebEngine ou MSS peuvent fournir
moins d'images selon la charge du système.

Le bouton **Ouvrir une image** accepte PNG/JPEG/BMP. La photo est réduite de
moitié, puis les deux méthodes reconstruisent la taille originale. L'interface
affiche PSNR et SSIM pour chaque résultat, calculés avec la même référence.
**Afficher les différences** remplace temporairement le panneau ESPCN par
`|ESPCN - Bicubique|` avec un contraste amplifié; cette carte ne prouve pas
qu'une méthode est meilleure. La molette sur l'un des deux panneaux applique
le même zoom aux deux images; **Réinitialiser le zoom** revient à 1x.
Pour vérifier que les deux sorties ne sont pas confondues, l'image fixe affiche
aussi leur MAE, leur différence maximale et le pourcentage de pixels différents.

**Comparaison agrandie** masque la zone du jeu ou de l'image originale et
donne presque tout l'espace vertical aux deux résultats. Le séparateur
vertical reste déplaçable. En mode normal, le jeu/image original occupe
environ 35 % et les résultats environ 65 %.

CUDA est obligatoire pour ESPCN : l'application affiche une erreur explicite
si `CUDAExecutionProvider` n'est pas disponible et ne bascule pas vers le CPU.
Les paquets NVIDIA listés dans `requirements.txt` fournissent les bibliothèques
compatibles avec ONNX Runtime GPU; le pilote NVIDIA du système reste requis.
F11 active ou désactive le plein écran. Le séparateur vertical peut être déplacé.

## U-Net de l'équipe

La branche `origin/ajout-unet` a été inspectée. Elle fournit
`Existing_Models/UNet/unet.py` et `eval.py`, conservés sans modification dans
ce dépôt. Elle ne contient actuellement aucun checkpoint `.pt` ou `.pth`.
L'adaptateur `unet_adapter.py` charge `models/unet_x4.pt` lorsqu'il est fourni;
sans ce fichier, le U-Net n'est volontairement pas proposé dans la fenêtre,
afin de ne pas présenter un modèle non fonctionnel. Le code original indique
un facteur x4 uniquement.

## Bicubique et ESPCN

Bicubique agrandit l'image par interpolation mathématique des pixels voisins.
ESPCN est un réseau convolutionnel pré-entraîné qui reconstruit une image haute
résolution à partir d'une image basse résolution.

Les FPS affichés sont le débit commun de lots effectivement terminés; les
millisecondes sont mesurées séparément pour chaque méthode. La barre
d'information distingue FPS de capture, FPS de comparaison, capture,
réduction, inférence et temps total. Chaque traitement possède son propre worker et son propre débit. Un seul lot
par méthode est en vol : si ESPCN est occupé, seule la dernière image est
conservée; Bicubique continue sans l'attendre. Les panneaux affichent donc
leurs FPS réellement affichés séparément, ainsi que le FPS réellement capturé.

Le bouton **Pause / Comparer la même image** arrête la capture sur la dernière
image disponible. Les deux méthodes traitent alors cette même image pour une
comparaison visuelle équitable. Le bouton **Reprendre le temps réel** réactive
la capture indépendante.

Sur l'environnement de développement, mesures CUDA à chaud sur une image
noire (temps d'inférence ESPCN; les FPS théoriques sont `1000 / total`) :

| Entrée | Bicubique | ESPCN CUDA | Total traitement | FPS théorique |
|---|---:|---:|---:|---:|
| 320 × 180 | 0,56 ms | 4,64 ms | 5,35 ms | 187,0 |
| 480 × 270 | 1,75 ms | 11,89 ms | 13,69 ms | 73,0 |
| 640 × 360 | 1,05 ms | 20,23 ms | 22,30 ms | 44,8 |

Ces médianes sont un micro-benchmark du traitement après échauffement, pas une
garantie de FPS réel dans le jeu. Le FPS réel dépend aussi de la capture, de
WebGL/MSS, de Qt et de la charge GPU.

À la fermeture, la capture est arrêtée avant les workers, les tâches en attente
sont annulées et les tâches déjà en cours sont attendues brièvement par Qt.
Les `CancelledError` attendues sont ignorées et aucune nouvelle tâche ne peut
être créée pendant l'arrêt.

## Limites

Au démarrage, la zone originale affiche simplement « Ouvrez un jeu, une vidéo
ou une image pour commencer. ». Il ne s'agit pas d'une mesure de jeu.
Le chargement d'un jeu dépend de la connexion réseau et de WebGL.
L'accès CUDA dépend du pilote NVIDIA et des bibliothèques installées. La
capture est volontairement limitée au widget du navigateur, afin de ne jamais
recapturer l'interface Sup-Eirb.
