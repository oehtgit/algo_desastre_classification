# Génération d'images satellites « avec désastre » — DeepGlobe

Ce projet prend une image satellite **saine** du dataset DeepGlobe et fabrique, **par des algorithmes classiques de traitement d'image** (aucun réseau de neurones, aucune donnée de catastrophe réelle), la même image après :

| Désastre | Clé | Résumé visuel |
|---|---|---|
| Inondation | `flood` | eau boueuse qui déborde des rivières/canaux, toits et arbres qui émergent |
| Incendie | `fire` | végétation carbonisée, maisons réduites en cendres, front de flammes et fumée (parfois) |
| Séisme | `earthquake` | bâtiments effondrés en gravats autour d'un épicentre |
| Tornade | `tornado` | couloir étroit : maisons arrachées, arbres couchés, sol décapé, débris dispersés |

Pour chaque image générée, le script produit aussi un **masque binaire des dégâts** et un **label**. L'objectif est de constituer un jeu de données pour entraîner/tester un modèle de **classification** (`none` / `flood` / `fire` / `earthquake` / `tornado`) ou de **segmentation des dégâts**.

> **Nouveau : images de drone à basse altitude.** Le script `algo_desastre_drone.py` applique les mêmes simulateurs au *Semantic Drone Dataset* et produit des masques au style RescueNet. Voir la [section 9](#9-images-de-drone-à-basse-altitude--algo_desastre_dronepy).

---

## 1. Structure du projet

```
algo_desastre_classification/
├── algo_desastre_classification   # le script Python (sans extension)
├── README.md
├── dataset/                       # DeepGlobe (téléchargé depuis Kaggle)
│   ├── class_dict.csv
│   ├── metadata.csv
│   ├── train/  <id>_sat.jpg + <id>_mask.png   (803 paires)
│   ├── valid/  <id>_sat.jpg                   (171 images, sans masque)
│   └── test/   <id>_sat.jpg                   (172 images, sans masque)
├── sortie/                        # résultats en 1024 px
└── sortie_full/                   # résultats en pleine résolution
```

## 2. Le dataset : DeepGlobe Land Cover Classification

- Source : <https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset> (challenge DeepGlobe 2018).
- Images satellites RGB **2448 × 2448 px**, résolution **50 cm/pixel** (une tuile ≈ 1,2 km × 1,2 km).
- Les images de `train/` ont un **masque d'occupation du sol** en couleurs :

| Couleur (R,G,B) | Classe |
|---|---|
| (0,255,255) cyan | urban_land — zone urbaine |
| (255,255,0) jaune | agriculture_land — cultures |
| (255,0,255) magenta | rangeland — prairie / broussailles |
| (0,255,0) vert | forest_land — forêt |
| (0,0,255) bleu | water — eau |
| (255,255,255) blanc | barren_land — sol nu |
| (0,0,0) noir | unknown — inconnu |

Ces masques sont **grossiers** : ils annotent de grandes zones, pas des objets (dans un quartier « urbain », jardins, arbres et canaux sont aussi marqués « urbain »). C'est pourquoi le script refait sa propre analyse objet par objet (§ 4.1).

## 3. Installation et utilisation

Dépendances (versions testées) : Python 3.8, `numpy` 1.24, `opencv-python` 5.0, `scikit-image` 0.21.

```bash
pip install numpy opencv-python scikit-image
```

Le dataset (2,8 Go) n'est pas versionné dans ce dépôt. Pour le récupérer dans `dataset/` :

```bash
mkdir dataset
curl -L -o dataset/deepglobe.zip https://www.kaggle.com/api/v1/datasets/download/balraj98/deepglobe-land-cover-classification-dataset
python -c "import zipfile; zipfile.ZipFile('dataset/deepglobe.zip').extractall('dataset')"
rm dataset/deepglobe.zip
```

(ou téléchargement manuel depuis la page Kaggle, puis extraction dans `dataset/`).

### Une image, avec planches de visualisation

```bash
# les 4 désastres, image réduite à 1024 px (rapide)
python algo_desastre_classification --image dataset/train/435277_sat.jpg --type all --size 1024 --out sortie

# un seul désastre, pleine résolution (plus réaliste), résultat reproductible
python algo_desastre_classification --image dataset/train/707319_sat.jpg --type tornado --seed 2 --out sortie_full
```

Fichiers produits dans `--out` :

| Fichier | Contenu |
|---|---|
| `<id>_<désastre>.jpg` | image avec désastre |
| `<id>_<désastre>_damage.png` | masque des dégâts (blanc = endommagé) |
| `<id>_comparaison.jpg` | planche : original, classes DeepGlobe, **analyse du script**, puis pour chaque désastre l'image, le contour des dégâts et le masque |
| `<id>_zoom.jpg` | zoom pleine résolution original / désastre, pour juger le réalisme |

### Construire un dataset de classification

```bash
python algo_desastre_classification --dataset dataset/train --n 200 --size 1024 --out dataset_desastre
```

Résultat :

```
dataset_desastre/
├── none/        <id>.jpg               (image d'origine)
├── flood/       <id>_flood.jpg
├── fire/        <id>_fire.jpg
├── earthquake/  <id>_earthquake.jpg
├── tornado/     <id>_tornado.jpg
├── masks/       <id>_<désastre>.png
└── labels.csv   image, label, source, severity, damaged_ratio, damage_mask
```

### Options

| Option | Rôle |
|---|---|
| `--image` / `--dataset` | une image `*_sat.jpg`, ou un dossier entier |
| `--mask` | masque à utiliser (sinon `<id>_mask.png` est cherché automatiquement ; s'il n'existe pas, les classes sont estimées) |
| `--type` | `all`, `flood`, `fire`, `earthquake` ou `tornado` |
| `--severity` | intensité dans [0, 1] (par défaut tirée au hasard entre 0,2 et 0,5) |
| `--size` | redimensionne l'image (ex. 1024) pour aller plus vite |
| `--n` | mode dataset : nombre d'images source tirées au hasard |
| `--seed` | graine aléatoire, pour reproduire exactement un résultat |
| `--out` | dossier de sortie |

> **`severity` ≠ `damaged_ratio`.** `severity` est le *réglage* (part de la zone éligible touchée : zone inondable, zone combustible, part des toits qui s'effondrent, largeur de la tornade…). `damaged_ratio` est la *mesure* : part des pixels de l'image entière dans le masque des dégâts. Par exemple, un séisme de sévérité 0,5 sur une image surtout agricole ne donne que ~10 % de dégâts. C'est `damaged_ratio` qu'il faut utiliser pour qualifier les images.

## 4. Fonctionnement

Le traitement suit trois étapes, communes à tous les désastres :

```
image saine ─► 1. analyse de la scène ─► 2. zone touchée ─► 3. rendu objet par objet ─► image + masque des dégâts
```

### 4.1 Analyse de la scène (classe `Scene`)

1. **Échelle physique.** Tous les paramètres sont exprimés en mètres puis convertis en pixels (`ppm` = pixels par mètre, 2 en pleine résolution, ≈ 0,84 en 1024 px). Le rendu reste ainsi cohérent quelle que soit la taille choisie.
2. **Superpixels SLIC** (`skimage.segmentation.slic`) : l'image est découpée en régions homogènes d'environ 8 m (la taille d'une maison). Ces régions suivent les bords des objets réels et servent d'« objets » pour toute la suite.
3. **Végétation** : indice de verdure **ExG = (2G − R − B)/(R + G + B)** moyenné par superpixel, seuillé par la **méthode d'Otsu** (le seuil s'adapte à la teinte de chaque image).
4. **Toits** : parmi les régions non végétales de la zone urbaine, score de « contraste » = écart de luminance avec la médiane locale (filtre médian ≈ 40 m) + rougeur (canal a* de Lab, tuiles) + texture interne. Les 60 % les plus contrastées sont des toits, le reste est sol/route.
5. **Eau** : le masque DeepGlobe, complété **uniquement dans les zones « urbain »/« inconnu »** (où le masque rate canaux et bassins) par une détection dans l'image : régions **sombres, lisses et non rougeâtres**, avec un **seuillage à hystérésis** (on part des zones « sûres » et on ajoute les zones « probables » qui y sont connectées, ce qui suit un canal sur toute sa longueur). Hors ville on fait confiance au masque, car une forêt sombre ressemble à de l'eau vue de dessus.
6. **Statistiques photométriques** : niveau de bruit du capteur (écart absolu médian de la haute fréquence) et texture locale (luminance / luminance moyenne voisine), utilisés au rendu.

Le panneau **« analyse du script »** de la planche de comparaison affiche ce résultat (rouge = toit, vert = végétation, bleu = eau, beige = sol/route).

### 4.2 Outils communs

| Outil | Rôle |
|---|---|
| **Bruit fractal** (bruit de valeur multi-octaves, `noise_px`) | formes naturelles et irrégulières à une échelle donnée en mètres (relief, turbidité, débris…) |
| **Seuil par quantile** (`quantile_mask`) | toucher exactement la fraction `severity` de la zone éligible |
| **Accrochage aux superpixels** (`segmean`) | les dégâts suivent les objets (un jardin est inondé « en bloc ») |
| **Morphologie mathématique** | nettoyage des masques, débordements (dilatation), contours (érosion) |
| **Brouillage local** (`Scene.scramble`, `cv2.remap` avec un champ de déplacement aléatoire) | casser les structures nettes → gravats, débris, cendres, en gardant les couleurs d'origine |
| **Modulation par la texture** | changer la couleur d'une surface sans perdre ses détails |
| **Flou optique + bruit capteur** (`optics`, `finish`) | les zones synthétiques ont la même netteté et le même grain que la photo |

### 4.3 Inondation — `simulate_flood`

1. **Altitude fictive** : `0,75 × distance à l'eau + 0,21 × relief (≈ 250 m) + 0,04 × micro-relief`, avec un léger bonus pour forêt et sol nu. Sans eau visible, une **vallée fictive** traverse l'image.
2. Les pixels les plus bas (quantile `severity`) sont inondés ; **profondeur** = niveau − altitude.
3. **Couleur** : ocre boueux, plus sombre en eau profonde, + turbidité et reflets.
4. **Transparence selon l'objet** : sol/routes noyés ; arbres d'autant plus noyés que l'eau est profonde ; chaque maison reçoit une **hauteur aléatoire** et n'est submergée que si l'eau est plus profonde.
5. **Détails** : la rivière d'origine devient boueuse, ombre des toits et arbres sur l'eau, bande de sol mouillé plus sombre, laisse de débris au bord.

### 4.4 Incendie — `simulate_fire`

1. **Foyer** dans la végétation, **vent** de direction aléatoire : distance anisotrope (le feu va 2,2× plus loin sous le vent).
2. **Score de propagation** = distance + bruit (≈ 200 m et 20 m) + combustible (la végétation brûle plus facilement), accroché aux superpixels, puis seuil par quantile.
3. **Rendu** : arbres roussis en périphérie et carbonisés près du foyer ; sol et routes désaturés et assombris par la suie (texture conservée) ; maisons → empreinte de cendres grises, tôles rouillées, débris noirs, murs noircis ; **12 %** des maisons épargnées.
4. **Feu actif** (≈ 1 fois sur 3) : flammes sur le front sous le vent + panache de fumée (le front est translaté et flouté de proche en proche dans la direction du vent).

### 4.5 Séisme — `simulate_earthquake`

1. **Épicentre** sur un toit tiré au hasard.
2. **Vulnérabilité** de chaque toit : `0,55 × distance à l'épicentre + 0,30 × effet de site (bruit ≈ 80 m) + 0,15 × hasard individuel`. Les toits les plus vulnérables (fraction `severity`) **s'effondrent**, les suivants (0,6 × `severity`) sont **endommagés**.
3. **Bâtiment effondré** : emprise dilatée (~1,2 m) et découpée par du bruit (débris qui débordent sur la rue), pixels **brouillés** sur ~2,5 m (couleurs d'origine conservées), désaturation légère, **blocs de béton** clairs et **vides** sombres.
4. **Toit endommagé** : taches de ~1 m brouillées et assombries (trous, tôles arrachées).
5. Voile de poussière très léger autour des effondrements.

### 4.6 Tornade — `simulate_tornado`

Ce qui la distingue du séisme (d'après les relevés du National Weather Service et des images satellites de tornades réelles) :

| | Séisme | Tornade |
|---|---|---|
| Forme | zone étendue autour d'un épicentre | **couloir étroit et long**, bords nets, parfois **interrompu** |
| Bâtiments | s'effondrent **sur place** | **toits arrachés** ou maisons **balayées jusqu'à la dalle**, débris **dispersés** |
| Végétation | intacte | arbres **cassés, couchés en motif convergent** vers l'axe |
| Sol | intact | **décapé** (cicatrice brun clair) |
| Après quelques jours | — | **bâches bleues** sur les toits |

Algorithme :
1. **Trajectoire** : droite passant par une zone bâtie centrale, direction aléatoire, rendue sinueuse ; **largeur variable** et **soulèvements** (coupures) par des profils 1D aléatoires.
2. **Intensité du vent** : maximale sur l'axe, plus forte du côté droit (rotation + déplacement), modulée par des tourbillons secondaires.
3. **Direction du vent au sol** = déplacement + composante vers l'axe → motif convergent.
4. **Rendu selon l'intensité de chaque objet** : toits balayés (> 0,7, dalle + débris), arrachés (0,42–0,7, plancher sombre), légers (0,18–0,42, tuiles manquantes) ; **bâches bleues** sur ~45 % des toits endommagés (une image sur deux) ; canopée détruite ; **troncs couchés** dessinés dans la direction du vent local ; sol décapé ; **débris** = morceaux de toits prélevés en amont et redéposés jusqu'à 25 m plus loin.

## 5. Tests effectués

| Test | Résultat |
|---|---|
| Image synthétique (forêt, cultures, ville, rivière + masque) | les deux modes fonctionnent, sorties et `labels.csv` corrects |
| `707319` (quartier dense, 99 % urbain, canaux non annotés) | canaux et douves détectés dans l'image ; l'inondation part des canaux ; séisme et incendie crédibles en zoom |
| `435277` (ville + rivière + forêt + cultures) | rivière suivie, forêt du sud correctement classée en végétation (corrigé : elle était prise pour de l'eau) ; tornade traversant ville et rivière avec une interruption |
| `100694` (campagne, aucune maison) | inondation et incendie corrects ; séisme quasi invisible (normal : pas de bâti) |
| `valid/105036` (sans masque) | fonctionne avec les classes estimées ; voir limites |
| Mode dataset, 3 images, `--size 1024` | 5 images + 4 masques par source, `labels.csv` complet |

**Temps de calcul mesurés** (CPU, Windows) :

| Cas | Durée |
|---|---|
| 1 image, 4 désastres, `--size 1024` | ≈ 9 s |
| 1 image, 4 désastres, pleine résolution 2448 px | ≈ 60 s |
| Mode dataset, `--size 1024` | ≈ 7 s par image source → ≈ 1 h 30 pour les 803 images de `train/` |

## 6. Limites actuelles

**Analyse de la scène**
- La détection des **toits** est une heuristique (contraste local) : une partie des routes est prise pour des toits, et en campagne des haies ou des bords de champs peuvent l'être aussi (le séisme « endommage » alors des haies).
- La détection de l'**eau** dans l'image confond eau calme et végétation sombre/lisse. En présence d'un masque, elle est limitée aux zones urbaines ; **sans masque** (`valid/`, `test/`) la confusion peut se produire, et les canaux couverts d'algues ne sont pas toujours détectés.
- Les classes **estimées** sans masque sont très approximatives (par ex. tout « cultures » sur `valid/105036`).

**Réalisme**
- Les dégâts suivent les superpixels : on voit parfois des **contours anguleux** (eau sur la canopée, maisons submergées).
- À `--size 1024`, une maison fait ~8 px et un tronc 1–2 px : gravats, troncs couchés et débris ne se voient bien qu'en **pleine résolution**.
- Le **feu traverse l'eau** : sa propagation est calculée à vol d'oiseau, sans obstacle.
- Il n'y a **pas de vrai relief** (pas de modèle numérique de terrain) : l'altitude de l'inondation est fictive.
- Sur une image rurale, la **tornade** se voit peu (peu d'objets à détruire), alors que son masque de dégâts couvre tout le couloir.
- Les couleurs ont été réglées à l'œil sur quelques images ; aucune comparaison quantitative avec de vraies images post-catastrophe (ex. dataset xBD) n'a été faite.

**Utilisation pour l'apprentissage**
- Un classifieur risque d'apprendre les **signatures de l'algorithme** (teinte de l'eau, motif des gravats) plutôt que de vraies catastrophes : il faut évaluer un modèle entraîné sur ces données avec de **vraies images** (ex. xBD).
- Le masque de dégâts est **binaire**, sans niveau de gravité par bâtiment.
- Le mode dataset produit les 4 désastres sur **toutes** les images, même quand le désastre n'a pas de sens (séisme en pleine campagne : `damaged_ratio` très faible). Filtrer avec `damaged_ratio` si besoin.

## 7. Pistes d'amélioration

- Propagation du feu par **distance géodésique** (contourner l'eau et les routes larges).
- Détection de l'eau par la **texture** (écart-type local, filtres de Gabor) plutôt que par la couleur.
- Ne générer un désastre que s'il est plausible (pas de séisme sans bâti, pas d'inondation loin de toute eau…).
- Masque de dégâts **multi-niveaux** (aucun / léger / majeur / détruit, comme xBD).
- Ajouter des catastrophes : **glissement de terrain**, **tsunami**, **coulée de lave**.

## 8. Sources

- DeepGlobe Land Cover Classification — Kaggle : <https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset>
- Achanta et al., *SLIC Superpixels*, IEEE TPAMI 2012 (implémentation scikit-image).
- Otsu, *A Threshold Selection Method from Gray-Level Histograms*, 1979.
- Gupta et al., *xBD: A Dataset for Assessing Building Damage from Satellite Imagery* : <https://arxiv.org/abs/1911.09296>
- NWS — motif convergent des dégâts de tornade : <https://www.weather.gov/lot/2009_08_19_tornadoes>
- Discontinuité et largeur des trajectoires (El Reno) : <https://ams.confex.com/ams/27SLS/webprogram/Manuscript/Paper254342/ElReno-FINAL.pdf>
- Images avant/après de tornades : Rolling Fork <https://foxweather.com/extreme-weather/rolling-fork-mississippi-tornado-before-after-photos>, Barnsdall (bâches bleues) <https://www.foxweather.com/extreme-weather/satellite-before-after-imagery-barsndall-oklahoma-tornado>

## 9. Images de drone — `algo_desastre_drone.py`

Le script gère deux sources d'images de drone **saines** (`--source`) :

| Source | Altitude | Usage |
|---|---|---|
| `vdd` (**par défaut**) — [VDD, Varied Drone Dataset](https://huggingface.co/datasets/RussRobin/VDD) | 50 à 120 m, DJI Mavic Air 2, 4000 × 3000 px | échelle proche des images RescueNet |
| `sdd` — Semantic Drone Dataset (TU Graz) | 5 à 30 m | très basse altitude (détaillé ci-dessous) |

### 9.1 VDD : vues verticales prises en altitude

- 400 images, 7 classes : `0` autre, `1` mur, `2` route, `3` végétation, `4` véhicule, `5` toit, `6` eau. Correspondance vérifiée visuellement sur les masques.
- Téléchargement : créer un compte Hugging Face, accepter les conditions sur la page du dataset, télécharger `VDD.zip` (2 Go) puis l'extraire dans `dataset_vdd/` (on obtient `dataset_vdd/VDD/{train,val,test}/{src,gt}`). Licence : usage non commercial, redistribution interdite.
- **Les photos n'ont aucune métadonnée** (ni EXIF, ni angle de nacelle, ni altitude). La sélection est donc faite à partir de l'image :
  - **vue à 90°** : au moins 3 % de toit, et murs/façades < 15 % de la surface bâtie (en vue oblique, les façades sont visibles) ;
  - **altitude estimée** = K / (pixels par mètre), avec une échelle tirée de la taille médiane des voitures (environ 8 m²) et K ≈ 2806, calculé d'après la caméra du Mavic Air 2. Cette estimation est **approximative** : quelques valeurs dépassent les 120 m annoncés, mais le classement relatif est cohérent à l'œil ;
  - seules les images **estimées à 85 m ou plus** sont retenues (`--min-alt`), soit **49 images**. Les vues verticales sans voiture visible sont écartées, faute d'échelle.

```bash
python algo_desastre_drone.py --list                                   # images retenues + altitude estimée
python algo_desastre_drone.py --image DJI_0716 --type all --seed 1     # une image -> sortie_vdd/
python algo_desastre_drone.py --dataset --out dataset_vdd_desastre     # les 49 images
```

Temps de calcul : environ 40 s par image pour les 4 désastres, à 2000 px de large. Les 49 images prennent environ 35 min.

### 9.1 bis UDD-6 : complément urbain (optionnel, `--source udd`)

- Source : <https://github.com/MarcWong/UDD> (Urban Drone Dataset, Université de Pékin). On utilise **UDD-6** : 141 images 4096 × 2160 prises à la verticale au-dessus de Pékin, Huludao, Cangzhou et Henan (campus, quartiers denses, rues). Licence : usage non commercial.
- Téléchargement : archive « UDD-5 + UDD-6 » (Google Drive, 1,2 Go), dont on extrait seulement `UDD/UDD6/` dans `dataset_udd/` (on obtient `dataset_udd/UDD/UDD6/{train,val}/{src,gt}`, 520 Mo).
- Les classes ont les **mêmes numéros que VDD** (0 other, 1 façade = wall, 2 road, 3 vegetation, 4 vehicle, 5 roof ; pas d'eau). Les masques sont en couleurs RGB : le profil `UDDProfile` les convertit à la lecture.
- Sélection : façades/(façades + toits) < 50 % (au lieu de 15 %, car les immeubles hauts montrent leurs façades en bord d'image même en vue verticale), pas de seuil d'altitude (caméra supposée, estimation peu fiable). Résultat : **105 images**.
- Limite : l'eau n'est pas annotée dans UDD (classée « other »), donc des débris peuvent tomber sur un lac.
- **VDD reste la source par défaut** : sans `--source udd`, rien ne change. Pour retirer UDD complètement : supprimer `dataset_udd/` et remplacer `algo_desastre_drone.py` par la sauvegarde `algo_desastre_drone_avant_udd.py`.

```bash
python algo_desastre_drone.py --source udd --list
python algo_desastre_drone.py --source udd --dataset --type tornado --out dataset_udd_desastre
```

### 9.2 Semantic Drone Dataset (basse altitude)

### Dataset : Semantic Drone Dataset (TU Graz)

- Source : <https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset>
- 400 images **saines** de quartiers résidentiels, prises **à la verticale** depuis un drone, **entre 5 et 30 m d'altitude**, en 6000 × 4000 px.
- Chaque image a un masque annoté pixel par pixel en **24 classes**, dont `roof`, `wall`, `paved-area`, `grass`, `tree`, `vegetation`, `water`, `pool` et `car`.
- RescueNet n'a pas été utilisé : ses images sont déjà prises *après* une catastrophe, alors que ce projet part d'images saines.

Téléchargement dans `dataset_drone/` (3,9 Go une fois extrait) :

```bash
mkdir dataset_drone
curl -L -o dataset_drone/semantic_drone.zip https://www.kaggle.com/api/v1/datasets/download/bulentsiyah/semantic-drone-dataset
python -c "import zipfile; z=zipfile.ZipFile('dataset_drone/semantic_drone.zip'); z.extractall('dataset_drone', [m for m in z.namelist() if not m.startswith('RGB_color_image_masks')])"
rm dataset_drone/semantic_drone.zip
```

### Sorties

Pour chaque image et chaque désastre :

| Fichier | Contenu |
|---|---|
| `<id>_sain.jpg` | image saine (redimensionnée, 1500 px de large par défaut) |
| `<id>_<désastre>.jpg` | image avec désastre |
| `<id>_<désastre>_masque.png` | masque au style RescueNet, 3 classes + fond |
| `<id>_<désastre>_planche.jpg` | image saine, image avec désastre et masque côte à côte (mode image seule) |

Couleurs du masque. Ce sont celles de RescueNet, d'après le fichier [`data/rescuenet.py`](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation/blob/main/Segmentation-Experiments/data/rescuenet.py) du dépôt officiel :

| Classe | Couleur (R,G,B) | Classe RescueNet d'origine |
|---|---|---|
| bâtiment intact | (180, 120, 120) | building-no-damage |
| bâtiment endommagé | (255, 0, 0) | building-total-destruction |
| route | (140, 140, 140) | road-clear |
| fond | (0, 0, 0) | unlabeled |

RescueNet distingue 4 niveaux de dégâts. Ici, ils sont regroupés en une seule classe « endommagé », affichée avec la couleur de *total-destruction*.

### Utilisation

```bash
# une image, les 4 désastres, avec planches
python algo_desastre_drone.py --source sdd --image 259 --type all --seed 1 --out sortie_drone

# tout le dataset (images avec au moins 5 % de toit), + labels.csv
python algo_desastre_drone.py --source sdd --dataset --out dataset_drone_desastre
```

Options : `--type`, `--severity`, `--seed`, `--width` (largeur de travail, 6000 = pleine résolution), `--n` (nombre d'images), `--min-roof` (part minimale de toit pour garder une image), `--root` (dossier `semantic_drone_dataset`).

### Différences avec la version satellite

- **La scène vient des annotations** au lieu d'être devinée par heuristique. Toit + mur + fenêtre + porte = bâtiment, et **chaque bâtiment (composante connexe) est un objet entier**. `paved-area` = route. Herbe, arbres et végétation = végétation. `water` = eau.
- **Échelle estimée par image** : l'altitude varie, donc le script estime le nombre de pixels par mètre à partir de la plus grande voiture annotée entièrement visible (environ 8 m² vue de dessus). Sans voiture, il prend une valeur par défaut de 250 px/m à 6000 px de large.
- **Paramètres plafonnés en pixels** : flou des bords ≤ 3 px, flou optique ≤ 1 px, rayons morphologiques ≤ 1 % de l'image, grain des gravats ≤ 2 px. Sans ces plafonds, les valeurs prévues en mètres pour le satellite rendaient les images de drone floues et « peintes ».
- **Règles propres au drone** : la chaussée ne brûle pas et n'est pas décapée par la tornade ; l'incendie part d'une maison ; pas de flammes ni de fumée (images post-événement, comme RescueNet).
- **Bâtiment endommagé** : des bâtiments collés forment un seul bloc, déclaré entièrement endommagé dès qu'il est touché, car RescueNet annote bâtiment par bâtiment.

### Tests

- Images 259 (rue et place) et 514 (maison et jardin) : les 4 désastres produisent image et masque.
- Mode dataset sur 2 images : 9 fichiers par image et `labels.csv` complet.
- Temps : environ 23 s par image pour les 4 désastres en 1500 px. Le filtre `--min-roof 0.05` retient 135 images, soit environ 50 min pour tout traiter.

### Limites (drone)

- **Le réalisme est nettement inférieur à la version satellite.** À environ 1 à 3 cm par pixel, on voit chaque tuile et chaque brin d'herbe, et le bruit procédural ne reproduit pas la texture de vrais gravats, de vraies cendres ou de vrais débris. La tornade (débris en aplats, troncs en traits uniformes) et les toits incendiés sont les cas les moins crédibles.
- **Toute l'image est souvent dans la zone touchée.** Une photo ne couvre qu'environ 20 à 60 m, alors qu'une tornade fait 100 m ou plus de large : le plus souvent, tous les bâtiments sont endommagés.
- **L'échelle par défaut est approximative** quand aucune voiture n'est visible.
- **Le masque ne suit pas toujours ce qu'on voit.** Un bloc touché, même très peu, est entièrement « endommagé », même si l'atteinte visible est petite.
- **Pas de classe « route bloquée »** : les routes recouvertes d'eau ou de débris restent « route », comme demandé (3 classes).
- **Le dataset est petit et peu varié** : 400 images, un seul quartier pavillonnaire autrichien.
- Vérifiez la licence du Semantic Drone Dataset avant de diffuser les images générées.
