# Compte rendu : tornade sur images de drone

📄 **[compte_rendu_tornade.pdf](compte_rendu_tornade.pdf)** (4 pages) résume le travail le plus récent :

- **Objectif :** générer des images de drone avec dégâts de tornade et leurs masques au style RescueNet (fond, route, bâtiment intact, bâtiment endommagé), pour compléter RescueNet lors de l'entraînement d'un modèle PSPNet.
- **Données :** 385 images saines déjà annotées, issues de six datasets publics (InstanceBuilding, Okutama, Semantic Drone Dataset, UDD, VDD, Swiss Drone), avec une comparaison de leur qualité et de leur résolution.
- **Méthode :** génération procédurale par étapes successives, textures réelles de dégâts de tornade (photos du domaine public ou sous CC0, synthétisées par *image quilting*), et intensité réglable de 0 à 1 : la gravité de chaque bâtiment touché augmente avec l'intensité.
- **Exemples, limites et reste à faire :** autres catastrophes (inondation, incendie, séisme), routes absentes des masques InstanceBuilding.

Pour générer une évolution de 0 à 1 sur une image (ici Okutama, 6 niveaux), comme dans le compte rendu :

```bash
python algo_desastre_drone.py --source okutama --image okutama_04_90_015 \
    --evolution 6 --seed 3 --textures-tornade --morceaux-naturels --out tests_intensite
```

Les datasets ne sont pas inclus dans le dépôt (plusieurs Go) : ils sont à télécharger séparément.
