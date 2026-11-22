# Journal de recherche : Watermark Localization

Tous les scores ci-dessous portent sur le train (510 images). Ce sont des diagnostics, pas des scores de test. Le split de validation commun est fixé dans `research/folds.json` : 5 plis stratifiés par méthode × polarité du fond × région minoritaire de moins ou de plus de 5 %.

## Étape 1 : signal gratuit des extracteurs officiels (21/11)

Commandes, depuis la racine avec le venv actif :

```bash
python -m research.zero_shot      # cartes zero-shot -> outputs/zero_shot/
python -m research.probes         # crop, gabarit, simulation -> outputs/probes/
python -m research.polarity_fill  # soumission de secours -> outputs/submissions/polarity_fill.pt
```

Résultats détaillés : [zero_shot.md](results/zero_shot.md), [probes.json](results/probes.json), [polarity_fill.json](results/polarity_fill.json).

### Polarité du fond : quasi résolue

La précision globale des bits par rapport au message connu sépare les fonds marqués (0,97 à 0,98) des fonds propres (0,53 à 0,55).

- **AUC pour `full_watermark` :** de 0,996 à 0,999.
- **Seuil par méthode, hors pli :** 95,9 % de bonnes polarités sur PixelSeal, 97,6 % sur TrustMark et 98,2 % sur StegaStamp.
- **mIoU hors pli de la soumission « remplir avec la polarité prédite » :** 0,451. Pour comparaison, tout propre donne 0,252.
- **Test :** l'histogramme de précision des bits est bimodal, avec 51 % de fonds marqués prédits.

Le fichier `outputs/submissions/polarity_fill.pt` est valide (1 020 masques, 16,7 Mo) et n'a pas été envoyé.

### Cartes locales zero-shot : faibles

| AUC moyenne par image | PixelSeal | TrustMark | StegaStamp |
|---|---:|---:|---:|
| Logits par position (résolution native) | 0,629 | 0,594 | — |
| Occlusion de blocs 32 px | 0,627 | 0,538 | 0,533 |
| Énergie haute fréquence | 0,548 | 0,565 | 0,607 |
| Gabarit `embed(I) − I` | 0,493 | 0,492 | **0,661** |
| Gradient × haute fréquence | 0,50 | 0,49 | 0,49 |

Le gradient × haute fréquence est au hasard et a été retiré du code. La localisation fine doit donc être apprise.

### Décodage du rectangle de la région marquée (fonds propres)

| Bit accuracy du crop | PixelSeal | TrustMark | StegaStamp |
|---|---:|---:|---:|
| Région exactement rectangulaire | 0,73 | **0,94** | 0,83 |
| Région irrégulière | 0,59 | 0,51 | 0,50 |
| Crop aléatoire de même taille | 0,54 | 0,51 | 0,50 |

Quand la région est un rectangle (outpainting), le crop redonne le message. Une recherche de rectangle qui maximise l'accord peut donc servir. Les régions irrégulières sont des objets transplantés depuis une image marquée en entier, et leur bounding box ne correspond pas au patch d'embedding.

### Fidélité de la simulation

J'ai fabriqué des compositions `where(M, embed(patch), X)` sur COCO, avec les vrais masques du train et le patch pris égal à la bounding box du masque.

- **Cartes zero-shot de PixelSeal :** elles ont la même AUC que sur les vraies images.
- **Crop des régions irrégulières :** il décode à 0,70–0,85 en simulation, contre environ 0,5 sur les vraies images. La simulation est donc fausse sur ce point : il faut marquer l'image source entière, ou un grand rectangle, puis découper l'objet.

### Nature des données (inspection visuelle)

- **Images :** elles sont générées par IA.
- **Fonds propres :** la région marquée est un objet de la scène ou un rectangle.
- **Fonds marqués :** les trous sont des objets ou des taches aléatoires, de type masque d'inpainting.

Les bords des masques suivent souvent ceux des objets.

## Étape 2 : segmenteurs entraînés, JURECA et ensembles (21/11 soir)

### Soumissions

| Manifeste | Contenu | mIoU fold 0 | Public (id) |
|---|---|---:|---:|
| `sub01_seg_v1` | premier segmenteur local | 0,580 | 0,5731 (1138) |
| `sub02_jureca` | un modèle long par méthode (JURECA) | 0,665 | 0,6502 (1233) |
| `sub03_single` | meilleur modèle par méthode | 0,684 | non envoyé |
| `sub03_ens` | ensembles de logits par méthode | 0,696 | 0,6878 (1400) |
| `sub04_folds` | `sub03_ens` + modèles entraînés sur les folds 1 et 2 | contaminé (0,885) | **0,6929 (1576)** |

Les modèles `--val-fold k` voient le fold 0 à l'entraînement : un ensemble mixte se juge sur le score public, pas sur le fold 0.

### Ce qui a marché, ce qui n'a rien donné

- **Gain :** entraînements longs (20 000 pas, batch 48) puis moyenne des logits de plusieurs checkpoints par méthode : +0,038 puis +0,005 en public.
- **Nul :** variantes de données (v2, v3 natif, frais, message) et branche pleine résolution `wmhr:` pour les petites régions.
- **Erreur restante :** surtout les petites régions (moins de 5 % de l'image). La polarité est résolue.

### Infrastructure

- `research.gpuq` : file locale partagée entre agents (GPU exclusif, verrous lecture/écriture sur les jeux de données).
- `research.upload` : envoi validé (sha256, refus d'un second envoi), score public enregistré dans le manifeste.
- Compte avert1 : droits Slurm retirés à 20:50. Le job 4 GPU lancé sous zhang96 (15903306, `heavy_f2` complet) a échoué en une minute : `setup_teammate.sh` relie l'environnement d'avert1, dont certains fichiers (`trustmark/models/trustmark_Q.yaml`) ne sont pas lisibles par zhang96. Il faut ouvrir ces droits au groupe ou un environnement propre à zhang96 avant de relancer.
