# Sup-Eirb

Sup-Eirb est une démonstration simple de comparaison de modèles de
super-résolution. Les modèles proposés doivent provenir de la branche
`corrections`, avoir leurs poids entraînés présents et disposer d'un
adaptateur validé. La même image basse résolution est alors envoyée aux deux
workers, puis les résultats sont affichés côte à côte.

```text
jeu choisi -> capture QtWebEngine ou MSS -> réduction 320/480/640
                                      +-> modèle 1 de corrections -> panneau gauche
                                      +-> modèle 2 de corrections -> panneau droit
```

## Fichiers

- `main.py` : lance l'application.
- `game.py` : navigateur intégré, clavier, souris, Pointer Lock et capture.
- `capture.py` : liste des fenêtres X11 et capture MSS de Firefox.
- `super_resolution.py` : inventaire strict et détection des modèles de
  `corrections`.
- `sequence.py` : tri numérique, lecture bornée et historique des séquences.
- `interface.py` : fenêtre redimensionnable, séparateur, mesures et plein écran.

## Installation et lancement

L'environnement `.venv` étant déjà fourni :

```bash
cd ~/Sup-Eirb
python -m pip install -r requirements.txt
python main.py
```

La fenêtre de configuration propose deux modèles et un facteur commun. Si
moins de deux modèles fonctionnels sont disponibles, elle affiche un message
explicite et ne permet pas de lancer une comparaison. La barre
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
**Afficher les différences** remplace temporairement le second panneau par la
différence amplifiée entre les deux modèles sélectionnés; cette carte ne prouve
pas qu'une méthode est meilleure. La molette sur l'un des deux panneaux applique
le même zoom aux deux images; **Réinitialiser le zoom** revient à 1x.
Pour vérifier que les deux sorties ne sont pas confondues, l'image fixe affiche
aussi leur MAE, leur différence maximale et le pourcentage de pixels différents.

**Comparaison agrandie** masque la zone du jeu ou de l'image originale et
donne presque tout l'espace vertical aux deux résultats. Le séparateur
vertical reste déplaçable. En mode normal, le jeu/image original occupe
environ 35 % et les résultats environ 65 %.

Chaque adaptateur devra déclarer explicitement son périphérique et ses facteurs
compatibles; aucune conversion automatique d'un facteur ou aucun repli
silencieux vers un autre modèle n'est autorisé.
Les paquets NVIDIA listés dans `requirements.txt` fournissent les bibliothèques
compatibles avec ONNX Runtime GPU; le pilote NVIDIA du système reste requis.
F11 active ou désactive le plein écran. Le séparateur vertical peut être déplacé.

## État de la branche corrections

`origin/corrections` contient les familles EDSR, SPAN, Real-ESRGAN et UNet.
Elle contient actuellement du code, des configurations et des dépendances,
mais aucun fichier de poids `.pth`, `.pt`, `.onnx`, `.ckpt` ou
`.safetensors`. Aucun modèle n'est donc déclaré utilisable par l'application.
Les anciens modèles locaux ne sont volontairement plus proposés.

Les facteurs indiqués dans les configurations sont EDSR x2/x3/x4, SPAN
x2/x3/x4, Real-ESRGAN x4 et UNet x2/x3/x4. Ils ne seront proposés qu'après
ajout des poids correspondants et d'un adaptateur d'inférence testé.

Les FPS affichés sont le débit de lots effectivement terminés; les
millisecondes sont mesurées séparément pour chaque méthode. La barre
d'information distingue FPS de capture, FPS de comparaison, capture,
réduction, inférence et temps total. Chaque traitement possède son propre worker et son propre débit. Un seul lot
par modèle est en vol : si un modèle est occupé, seule la dernière image est
conservée; l'autre continue sans l'attendre. Les panneaux affichent donc
leurs FPS réellement affichés séparément, ainsi que le FPS réellement capturé.

Le bouton **Pause / Comparer la même image** arrête la capture sur la dernière
image disponible. Les deux méthodes traitent alors cette même image pour une
comparaison visuelle équitable. Le bouton **Reprendre le temps réel** réactive
la capture indépendante.

Les FPS réels dépendent de la capture, de WebGL/MSS, de Qt, du modèle et de la
charge GPU. Aucun benchmark n'est annoncé tant qu'un modèle de `corrections`
n'est pas effectivement chargé.

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
