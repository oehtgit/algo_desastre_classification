# Textures réelles de dégâts de tornade

Utilisées uniquement avec l'option `--textures-tornade` de `algo_desastre_drone.py`.

Échantillons découpés dans des **photos aériennes réelles de tornades**, trouvées sur Wikimedia Commons et signalées dans le domaine public ou sous CC0 (licence, auteur et page de chaque photo dans `sources/sources.txt`). Aucune image ne provient de RescueNet ni d'un autre jeu de données d'entraînement.

| Échantillon | Contenu | Photo source | Échelle source estimée |
|---|---|---|---|
| `debris_1.png` à `debris_5.png` | gravats de maisons détruites | `sources/t02.jpg` — tornade EF4 de Greenfield (Iowa), CC0 | ~48 px/m |
| `toit_1.png` à `toit_4.png` | charpentes et voliges à nu | `sources/t15.jpg` — tornade de Moore (Oklahoma), 2013, domaine public | ~23 px/m |

Les échantillons sont ramenés à environ 16 px/m, l'échelle attendue par le script. Les échelles sont estimées à partir de la taille des voitures, donc approximatives, car les vues sont un peu obliques.
