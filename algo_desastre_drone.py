# -*- coding: utf-8 -*-
"""
algo_desastre_drone.py
======================

Génération d'images de drone "avec désastre" à partir d'images SAINES, avec un
masque au style RescueNet. Deux sources possibles :

  vdd (défaut)  VDD - Varied Drone Dataset (https://huggingface.co/datasets/RussRobin/VDD)
                400 images 4000 x 3000 px, DJI Mavic Air 2, 50 à 120 m d'altitude,
                vues à 30, 60 ou 90 degrés, 7 classes. On ne garde que les vues
                VERTICALES (90°) avec bâtiments et une altitude estimée élevée,
                proche de celle des images RescueNet.
  sdd           Semantic Drone Dataset (TU Graz), 5 à 30 m d'altitude
                (https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)

Sorties pour chaque image :
    <id>_sain.jpg                image saine (redimensionnée)
    <id>_<desastre>.jpg          image avec désastre
    <id>_<desastre>_masque.png   masque style RescueNet : fond (0,0,0),
                                 bâtiment intact (180,120,120),
                                 bâtiment endommagé (255,0,0), route (140,140,140)
    <id>_<desastre>_planche.jpg  saine | désastre | masque (mode image seule)

Usage :
    python algo_desastre_drone.py --list                       # images VDD retenues
    python algo_desastre_drone.py --image DJI_0739 --type all  # une image
    python algo_desastre_drone.py --dataset --out dataset_vdd_desastre
    python algo_desastre_drone.py --source sdd --image 259
"""

import argparse
import csv
import json
import importlib.machinery
import importlib.util
import os
import sys
from glob import glob

import cv2
import numpy as np
from skimage.segmentation import slic

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_core():
    """Charge le moteur de simulation (fichier sans extension .py)."""
    path = os.path.join(HERE, "algo_desastre_classification")
    loader = importlib.machinery.SourceFileLoader("algo_desastre_core", path)
    spec = importlib.util.spec_from_loader("algo_desastre_core", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    loader.exec_module(mod)
    return mod


core = _load_core()
# à quelques cm/pixel, des bords flous de plusieurs dizaines de cm donnent un
# aspect "peinture" : on plafonne l'adoucissement des bords à 3 px
core.SOFT_CAP_PX = 3.0

# ---------------------------------------------------------------------------
# Textures RÉELLES de gravats (photos aériennes du séisme de 2023, domaine
# public, voir textures_reelles/SOURCES.md) -> synthèse de texture par
# "image quilting" (Efros & Freeman, SIGGRAPH 2001)
# ---------------------------------------------------------------------------
TEX_DIR = os.path.join(HERE, "textures_reelles")
TEX_SRC_PPM = 16.0          # échelle estimée des échantillons (px/m)
USE_REAL_TEX = True
_DEFAULT_TINT = (150, 142, 130)
# Textures de TORNADE (option --textures-tornade, voir textures_tornade/SOURCES.md) :
# sans l'option, le rendu reste exactement celui des gravats ci-dessus
TORNADO_TEX_DIR = os.path.join(HERE, "textures_tornade")
USE_TORNADO_TEX = False
_ACTIVE_TEX = "gravats"     # jeu de textures du lit de gravats pour la simulation en cours
TEX_SETS = {"gravats": (TEX_DIR, "gravats_*.png"),
            "tornade": (TORNADO_TEX_DIR, "debris_*.png"),
            "toit": (TORNADO_TEX_DIR, "toit_*.png")}


def _flatten(img):
    """Supprime l'éclairage à grande échelle (soleil rasant, ombres douces)
    en gardant la texture fine : L <- L - flou(L) + moyenne."""
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    lab[..., 0] = L - cv2.GaussianBlur(L, (0, 0), max(4.0, L.shape[0] / 6)) + L.mean()
    return lab


def _load_tex_samples(ppm, name="gravats"):
    from glob import glob as _g
    out = []
    folder, pattern = TEX_SETS[name]
    for p in sorted(_g(os.path.join(folder, pattern))):
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
        s = ppm / TEX_SRC_PPM
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        out.append(_flatten(im))
    return out


def _min_cut(err):
    """Chemin de coût minimal (de haut en bas) dans err (H x W) ; renvoie,
    pour chaque ligne, la colonne de coupure."""
    H, W = err.shape
    E = err.copy()
    for y in range(1, H):
        left = np.r_[np.inf, E[y - 1, :-1]]
        right = np.r_[E[y - 1, 1:], np.inf]
        E[y] += np.minimum(np.minimum(left, E[y - 1]), right)
    path = np.zeros(H, int)
    path[-1] = int(np.argmin(E[-1]))
    for y in range(H - 2, -1, -1):
        c = path[y + 1]
        lo, hi = max(0, c - 1), min(W, c + 2)
        path[y] = lo + int(np.argmin(E[y, lo:hi]))
    return path


def quilt(samples, h, w, rng, B=40, O=8, ncand=60):
    """Synthèse de texture par patchs (image quilting) : patchs tirés dans
    les échantillons réels, choisis pour se raccorder aux voisins, joints
    par une découpe de coût minimal (pas de bord visible)."""
    step = B - O
    H, W = h + B, w + B
    out = np.zeros((H, W, 3), np.float32)
    usable = [s for s in samples if s.shape[0] > B and s.shape[1] > B]
    if not usable:
        return None
    for y in range(0, h, step):
        for x in range(0, w, step):
            si = rng.integers(len(usable), size=ncand)
            cand = []
            for k in si:
                s = usable[k]
                yy = rng.integers(0, s.shape[0] - B)
                xx = rng.integers(0, s.shape[1] - B)
                cand.append(s[yy:yy + B, xx:xx + B])
            cand = np.stack(cand)
            cost = np.zeros(ncand)
            if x > 0:
                cost += ((cand[:, :, :O] - out[y:y + B, x:x + O]) ** 2).sum((1, 2, 3))
            if y > 0:
                cost += ((cand[:, :O, :] - out[y:y + O, x:x + B]) ** 2).sum((1, 2, 3))
            best = np.nonzero(cost <= cost.min() * 1.1 + 1e-6)[0]
            patch = cand[rng.choice(best)]
            mask = np.ones((B, B), bool)
            if x > 0:
                e = ((patch[:, :O] - out[y:y + B, x:x + O]) ** 2).sum(2)
                cut = _min_cut(e)
                for r in range(B):
                    mask[r, :cut[r]] = False
            if y > 0:
                e = ((patch[:O, :] - out[y:y + O, x:x + B]) ** 2).sum(2).T
                cut = _min_cut(e)
                for c in range(B):
                    mask[:cut[c], c] = False
            reg = out[y:y + B, x:x + B]
            out[y:y + B, x:x + B] = np.where(mask[..., None], patch, reg)
    return out[:h, :w]


def _scene_rubble(sc, rng, name="gravats"):
    """Texture réelle couvrant toute l'image, harmonisée avec la scène
    (luminosité, contraste, teinte), calculée une fois par scène et par jeu."""
    cache = sc.__dict__.setdefault("_tex_cache", {})
    if name in cache:
        return cache[name]
    samples = _load_tex_samples(sc.ppm, name)
    # échantillons de toits plus petits (charpente : motif fin)
    B = int(np.clip(sc.m(1.5), 12, 32)) if name == "toit" else int(np.clip(sc.m(3.0), 24, 64))
    # graine propre (ne consomme pas le générateur principal : même géométrie
    # de dégâts avec ou sans textures réelles, pour comparer)
    lab = quilt(samples, sc.h, sc.w, np.random.default_rng(sc.h * 7919 + sc.w + (0 if name == "gravats" else len(name))),
                B=B, O=max(4, B // 5))
    if lab is None:
        cache[name] = False
        return False
    # statistiques des surfaces MINÉRALES (bâti + sol), pas de la végétation
    # (sinon les gravats prennent une teinte verte dans les quartiers boisés)
    img_lab = cv2.cvtColor(sc.img, cv2.COLOR_RGB2LAB).astype(np.float32)
    mineral = getattr(sc, "roof", None)
    if mineral is not None:
        mineral = (sc.roof | sc.ground) & ~sc.veg
    sel = img_lab[mineral] if mineral is not None and mineral.sum() > 1000 else img_lab.reshape(-1, 3)
    im_m, im_s = sel.mean(0), sel.std(0)
    t_m, t_s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-3
    # gravats : un peu plus clairs que la scène (béton, poussière), contraste
    # de la scène, teinte rapprochée de celle de l'image (soleil rasant -> neutre)
    if name == "gravats":
        lab[..., 0] = (lab[..., 0] - t_m[0]) / t_s[0] * (0.85 * im_s[0]) + (im_m[0] + 14)
        lab[..., 1:] = (lab[..., 1:] - t_m[1:]) * 0.55 + 0.15 * t_m[1:] + 0.85 * im_m[1:]   # soleil rasant -> neutre
    else:
        # tornade : on garde les vraies couleurs (bois, plâtre, isolant)
        lab[..., 0] = (lab[..., 0] - t_m[0]) * 0.9 + 0.5 * t_m[0] + 0.5 * (im_m[0] + 10)
        lab[..., 1:] = (lab[..., 1:] - t_m[1:]) * 0.9 + 0.75 * t_m[1:] + 0.25 * im_m[1:]
    rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    cache[name] = rgb
    return rgb


def _install_real_rubble():
    """Remplace le lit de gravats procédural (bruit) par la texture réelle."""
    procedural = core.rubble_base

    def rubble_base(sc, rng, src, tint=_DEFAULT_TINT):
        base = procedural(sc, rng, src, tint)    # toujours appelé : mêmes tirages aléatoires
        if not (USE_REAL_TEX or _ACTIVE_TEX == "tornade") or tuple(tint) != _DEFAULT_TINT:
            return base
        tex = _scene_rubble(sc, rng, _ACTIVE_TEX)
        if tex is False:
            return base
        return tex.copy()

    core.rubble_base = rubble_base


_install_real_rubble()

# ---------------------------------------------------------------------------
# Morceaux de toit plus naturels (option --morceaux-naturels) : bords irréguliers,
# ni contour noir ni ombre franche. Sans l'option, le rendu d'origine est inchangé.
# ---------------------------------------------------------------------------
NATURAL_PIECES = False
_core_cells = core._cells
_core_draw_pieces = core.draw_pieces


def _nat_rng(sc):
    # générateur propre : ne change pas les tirages des dégâts
    if "_nat_rng" not in sc.__dict__:
        sc._nat_rng = np.random.default_rng(sc.h * 31 + sc.w + 1)
    return sc._nat_rng


def _natural_cells(sc, rng, region, cell_m, max_cells=60000):
    """Cellules de Voronoï aux bords déformés (cassures irrégulières)."""
    lab, n = _core_cells(sc, rng, region, cell_m, max_cells)
    if lab is None or not NATURAL_PIECES:
        return lab, n
    r = _nat_rng(sc)
    cell = max(2.0, sc.m(cell_m))
    amp = 0.35 * cell
    dx = (core.noise_px((sc.h, sc.w), r, max(2.0, 0.7 * cell), 2) - 0.5) * 2 * amp
    dy = (core.noise_px((sc.h, sc.w), r, max(2.0, 0.7 * cell), 2) - 0.5) * 2 * amp
    wob = cv2.remap(lab.astype(np.float32), (sc.xx + dx).astype(np.float32), (sc.yy + dy).astype(np.float32),
                    cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return wob.astype(np.int32), n


def _natural_draw_pieces(sc, out, pieces, valid, cracks, shadow_m=0.15, crack_dark=0.55):
    """Pose les morceaux avec un bord adouci, une ombre légère et des fissures
    discontinues, au lieu d'un contour sombre autour de chaque polygone."""
    if not NATURAL_PIECES or sc.m(0.4) < 3:
        return _core_draw_pieces(sc, out, pieces, valid, cracks, shadow_m, crack_dark)
    r = _nat_rng(sc)
    v = valid.astype(np.float32)
    soft = cv2.GaussianBlur(v, (0, 0), 0.7)[..., None]
    s = max(1.0, sc.m(shadow_m))
    shade = cv2.GaussianBlur(core.shift(v, core.SUN[0] * s, core.SUN[1] * s), (0, 0), max(0.8, 0.5 * s))
    shade = np.clip(shade - soft[..., 0], 0, 1)[..., None]
    out = out * (1 - 0.35 * shade)
    out = out * (1 - soft) + pieces * soft
    gaps = core.noise_px((sc.h, sc.w), r, max(3.0, sc.m(1.0)), 2) > 0.55     # fissures par tronçons
    c = cv2.GaussianBlur((cracks & gaps).astype(np.float32), (0, 0), 0.6)[..., None]
    return out * (1 - (1 - max(crack_dark, 0.82)) * c)


core._cells = _natural_cells
core.draw_pieces = _natural_draw_pieces

# Palette des masques de sortie (couleurs de RescueNet, voir README)
MASK_COLORS = {
    "fond": (0, 0, 0),
    "batiment_intact": (180, 120, 120),
    "batiment_endommage": (255, 0, 0),
    "route": (140, 140, 140),
}
CAR_AREA_M2 = 8.0          # empreinte d'une voiture vue de dessus (~4,5 m x 1,8 m)
DAMAGED_FRACTION = 1e-9    # > 0 : un bâtiment est endommagé dès qu'un seul de ses pixels est touché
TOUCH_GAP_M = 0.5          # demi-largeur des fentes (< ~1 m) entre deux bâtiments considérés collés
FRAGMENT_M2 = 30.0         # sous cette surface et sans toit : éclat d'annotation, pas un bâtiment
FRAGMENT_ROOF_M2 = 5.0     # surface de toit à partir de laquelle un petit morceau reste un bâtiment


# ---------------------------------------------------------------------------
# Profils de dataset
# ---------------------------------------------------------------------------
class SDDProfile:
    name = "sdd"
    root = os.path.join(HERE, "dataset_drone", "dataset", "semantic_drone_dataset")
    classes = ["unlabeled", "paved-area", "dirt", "grass", "gravel", "water", "rocks", "pool",
               "vegetation", "roof", "wall", "window", "door", "fence", "fence-pole", "person",
               "dog", "car", "bicycle", "tree", "bald-tree", "ar-marker", "obstacle", "conflicting"]
    C = {n: i for i, n in enumerate(classes)}
    building = [C["roof"], C["wall"], C["window"], C["door"]]
    road = [C["paved-area"]]
    veg = [C["grass"], C["vegetation"], C["tree"], C["bald-tree"]]
    grass = [C["grass"]]
    bare = [C["dirt"], C["gravel"], C["rocks"]]
    water = [C["water"]]
    pool = [C["pool"]]
    car = C["car"]
    default_ppm_full = 250.0       # px/m à pleine résolution si aucune voiture
    default_width = 1500

    def ids(self):
        return sorted(os.path.basename(p)[:-4]
                      for p in glob(os.path.join(self.root, "label_images_semantic", "*.png")))

    def paths(self, image_id):
        return (os.path.join(self.root, "original_images", "%s.jpg" % image_id),
                os.path.join(self.root, "label_images_semantic", "%s.png" % image_id))


class VDDProfile:
    name = "vdd"
    root = os.path.join(HERE, "dataset_vdd", "VDD")
    # valeurs des masques gt/ (vérifiées visuellement sur les images)
    C = {"other": 0, "wall": 1, "road": 2, "vegetation": 3, "vehicle": 4, "roof": 5, "water": 6}
    building = [C["roof"], C["wall"]]
    road = [C["road"]]
    veg = [C["vegetation"]]
    grass = []
    bare = []
    water = [C["water"]]
    pool = []
    car = C["vehicle"]
    default_ppm_full = 28.0
    default_width = 2000
    # Altitude estimée = K / (px par mètre à pleine résolution). K vient de la
    # caméra du DJI Mavic Air 2 (capteur 1/2" ~6,4 mm de large, focale ~4,5 mm,
    # 4000 px) : GSD ~ 3,56e-4 x altitude. Estimation APPROXIMATIVE.
    alt_k = 4000 * 4.49 / 6.4
    full_width = 4000
    max_wall_ratio = 0.15            # façade / (façade + toit) max pour une vue verticale
    default_min_alt = 85.0

    def read_label(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def _index(self):
        if not hasattr(self, "_idx"):
            self._idx = {}
            for gt in glob(os.path.join(self.root, "*", "gt", "*.png")):
                split = os.path.basename(os.path.dirname(os.path.dirname(gt)))
                self._idx[os.path.splitext(os.path.basename(gt))[0]] = split
        return self._idx

    def ids(self):
        return sorted(self._index())

    def paths(self, image_id):
        split = self._index()[image_id]
        return (os.path.join(self.root, split, "src", "%s.JPG" % image_id),
                os.path.join(self.root, split, "gt", "%s.png" % image_id))

    def selection(self, min_alt):
        """Vues verticales (presque pas de façades visibles) avec bâtiments.
        - avec voitures : altitude estimée, gardées si >= min_alt ;
        - sans voiture : altitude inconnue, gardées (VDD est pris entre 50 et
          120 m) ; on leur attribue l'échelle médiane des vues verticales
          mesurées (stockée dans default_ppm_full).
        Renvoie [(id, altitude ou None, px/m ou None)]."""
        measured, no_car = [], []
        for image_id in self.ids():
            lab = self.read_label(self.paths(image_id)[1])[::2, ::2]
            roof = (lab == self.C["roof"]).mean()
            wall = (lab == self.C["wall"]).mean()
            if roof < 0.03 or wall / (wall + roof) >= self.max_wall_ratio:
                continue
            ppm, src = estimate_ppm(lab, self.car, None)
            if src != "voiture":
                no_car.append((image_id, None, None))
                continue
            ppm *= 2                                  # masque sous-échantillonné de moitié
            measured.append((image_id, self.alt_k / ppm, ppm))
        if measured:
            self.default_ppm_full = float(np.median([t[2] for t in measured]))
        kept = sorted([t for t in measured if t[1] >= min_alt], key=lambda t: -t[1])
        return kept + no_car


class UDDProfile(VDDProfile):
    """Urban Drone Dataset, version UDD-6 (github.com/MarcWong/UDD) : vues
    verticales de villes chinoises, 4096 x 2160. Mêmes numéros de classes que
    VDD (0..5, pas d'eau) mais masques en couleurs RGB, convertis ici.
    Pour revenir à VDD seul : ne pas passer --source udd (VDD reste le défaut)."""
    name = "udd"
    root = os.path.join(HERE, "dataset_udd", "UDD", "UDD6")
    COLORS = {(0, 0, 0): 0, (102, 102, 156): 1, (128, 64, 128): 2,
              (107, 142, 35): 3, (0, 0, 142): 4, (70, 70, 70): 5}     # RGB -> classe
    water = []
    default_ppm_full = 28.0          # recalculé par selection() à partir des voitures
    # Caméra DJI Phantom 4 en vidéo 4K (capteur 1/2,3" ~6,17 mm, focale 3,61 mm,
    # 4096 px) : estimation APPROXIMATIVE de l'altitude.
    alt_k = 4096 * 3.61 / 6.17
    full_width = 4096
    # Immeubles hauts + grand angle : les façades apparaissent en bord d'image
    # même en vue verticale (rapport médian ~0,38), d'où un seuil plus large.
    # L'altitude estimée n'est pas fiable (caméra supposée) : pas de seuil.
    max_wall_ratio = 0.5
    default_min_alt = 0.0

    def read_label(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        lab = np.zeros(bgr.shape[:2], np.uint8)
        for (r, g, b), k in self.COLORS.items():
            lab[(bgr[..., 2] == r) & (bgr[..., 1] == g) & (bgr[..., 0] == b)] = k
        return lab


class InstanceBuildingProfile:
    """InstanceBuilding 2D (californiachen.github.io/datasets/InstanceBuilding) :
    un polygone VIA par toit, donc les bâtiments sont déjà individualisés.
    Seules les vues verticales *-OR.png (dalles 1000 x 1000) sont retenues.
    Pas de classe route ni végétation : la végétation est estimée par la
    couleur, et le masque de sortie ne contient aucune route."""
    name = "instancebuilding"
    root = os.path.join(HERE, "nouveau_dataset_2")
    C = VDDProfile.C
    building = [C["roof"]]
    road = [C["road"]]
    veg = [C["vegetation"]]
    grass = []
    bare = []
    water = [C["water"]]
    pool = []
    car = C["vehicle"]
    default_ppm_full = 9.0           # mesuré à l'œil sur les voitures (pas de classe véhicule)
    default_width = 1000
    full_width = 1000
    min_polygons = 5                 # images retenues : au moins ce nombre de toits

    def _index(self):
        if not hasattr(self, "_idx"):
            self._idx = {}
            jp = glob(os.path.join(self.root, "*", "2D", "via_region_data-train.json"))[0]
            with open(jp, encoding="utf-8") as fh:
                data = json.load(fh)
            tdir = glob(os.path.join(self.root, "train-*", "train"))[0]
            for v in data.values():
                path = os.path.join(tdir, v["filename"])
                regs = v["regions"] if isinstance(v["regions"], list) else list(v["regions"].values())
                if v["filename"].endswith("-OR.png") and len(regs) >= self.min_polygons and os.path.exists(path):
                    polys = [np.array(list(zip(r["shape_attributes"]["all_points_x"],
                                               r["shape_attributes"]["all_points_y"])), np.int32) for r in regs]
                    self._idx[v["filename"][:-4]] = (path, [p for p in polys if len(p) >= 3])
        return self._idx

    def ids(self):
        return sorted(self._index())

    def paths(self, image_id):
        return self._index()[image_id][0], image_id

    def _rasterize(self, image_id, shape):
        polys = self._index()[image_id][1]
        inst = np.zeros(shape, np.int32)
        # les plus grands d'abord : un petit toit reste visible en cas de recouvrement
        for k, p in enumerate(sorted(polys, key=lambda q: -abs(cv2.contourArea(q))), 1):
            cv2.fillPoly(inst, [p], k)
        _, new = np.unique(inst, return_inverse=True)      # numéros consécutifs (0 = fond)
        return new.reshape(shape).astype(np.int32)

    def read_label(self, image_id):
        bgr = cv2.imread(self.paths(image_id)[0], cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        inst = self._rasterize(image_id, bgr.shape[:2])
        b, g, r = cv2.split(cv2.GaussianBlur(bgr, (5, 5), 0).astype(np.float32))
        lab = np.full(inst.shape, self.C["other"], np.uint8)
        lab[2 * g - r - b > 20] = self.C["vegetation"]
        lab[inst > 0] = self.C["roof"]
        return lab

    def instance_map(self, image_id, shape):
        h0, w0 = cv2.imread(self.paths(image_id)[0], cv2.IMREAD_COLOR).shape[:2]
        inst = self._rasterize(image_id, (h0, w0))
        if (h0, w0) != tuple(shape):
            inst = cv2.resize(inst.astype(np.float32), (shape[1], shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return inst


class OkutamaProfile:
    """Okutama + Swiss Drone (okutama-segmentation.org), vues verticales annotées
    à la main. Une seule classe « Buildings » (pas de toit / mur), et « Paved
    ground » regroupe routes, parkings et trottoirs : la classe route du masque
    de sortie contient donc tout le sol revêtu."""
    name = "okutama"
    root = os.path.join(HERE, "nouveau_dataset_3", "extrait")
    C = {"background": 0, "outdoor": 1, "roof": 2, "paved": 3, "unpaved": 4, "tracks": 5,
         "plants": 6, "vehicle": 7, "water": 8, "people": 9}
    building = [C["roof"]]
    road = [C["paved"]]
    veg = [C["plants"]]
    grass = []
    bare = [C["unpaved"]]
    water = [C["water"]]
    pool = []
    car = C["vehicle"]
    default_ppm_full = 27.0          # estimation à 90 m ; remplacée par les voitures si présentes
    default_width = 2000
    full_width = 3840

    def _index(self):
        if not hasattr(self, "_idx"):
            self._idx = {}
            for gt in glob(os.path.join(self.root, "ground_truth", "*", "*.png")):
                image_id = os.path.splitext(os.path.basename(gt))[0]
                split = os.path.basename(os.path.dirname(gt))
                for ext in (".png", ".JPG"):
                    im = os.path.join(self.root, "images", split, image_id + ext)
                    if os.path.exists(im):
                        self._idx[image_id] = (im, gt)
        return self._idx

    def ids(self):
        return sorted(self._index())

    def paths(self, image_id):
        return self._index()[image_id]

    def read_label(self, path):
        lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if lab is not None and lab.ndim == 3:       # Okutama : 3 canaux identiques
            lab = lab[..., 0]
        return lab


PROFILES = {"vdd": VDDProfile, "sdd": SDDProfile, "udd": UDDProfile,
            "instancebuilding": InstanceBuildingProfile, "okutama": OkutamaProfile}


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def load_drone(prof, image_id, width):
    ip, lp = prof.paths(image_id)
    img = core.read_rgb(ip)
    lab = (prof.read_label(lp) if hasattr(prof, "read_label")
           else cv2.imread(lp, cv2.IMREAD_UNCHANGED))
    if lab is None:
        raise FileNotFoundError(lp)
    if width and img.shape[1] != width:
        h = int(round(img.shape[0] * width / img.shape[1]))
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab, (width, h), interpolation=cv2.INTER_NEAREST)
    return img, lab


def estimate_ppm(lab, car_id, default):
    """Pixels par mètre, d'après la médiane des voitures entièrement visibles
    (la plus grande pour une seule voiture)."""
    h, w = lab.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats((lab == car_id).astype(np.uint8), 8)
    areas = [stats[i, 4] for i in range(1, n)
             if stats[i, 0] > 0 and stats[i, 1] > 0 and stats[i, 0] + stats[i, 2] < w
             and stats[i, 1] + stats[i, 3] < h and stats[i, 4] > 20]
    if len(areas) >= 3:
        return float(np.sqrt(np.median(areas) / CAR_AREA_M2)), "voiture"
    if areas:
        return float(np.sqrt(max(areas) / CAR_AREA_M2)), "voiture"
    return default, "defaut"


# ---------------------------------------------------------------------------
# Scène à partir des annotations
# ---------------------------------------------------------------------------
def _full_footprints(bl, ppm):
    """Emprise de chaque bâtiment pour le masque : le contour annoté par VDD est
    conservé. On bouche seulement les fentes étroites (< ~60 cm) et les petits
    trous (< 2 m²) à l'intérieur d'un même bâtiment."""
    fp = bl.copy()
    h, w = bl.shape
    r = max(1, int(round(0.3 * ppm)))
    ker = core.disk(r)
    for i in range(1, int(bl.max()) + 1):
        ys, xs = np.nonzero(bl == i)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, ys.min() - 2 * r), min(h, ys.max() + 2 * r + 1)
        x0, x1 = max(0, xs.min() - 2 * r), min(w, xs.max() + 2 * r + 1)
        m = (bl[y0:y1, x0:x1] == i).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
        cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is not None:
            for c, hi in zip(cs, hier[0]):
                if hi[3] >= 0 and cv2.contourArea(c) < 2 * ppm ** 2:
                    cv2.drawContours(m, [c], -1, 1, -1)
        view = fp[y0:y1, x0:x1]
        view[(m > 0) & (view == 0)] = i
    return fp


class DroneScene(core.Scene):
    """Scène dont les objets viennent des annotations : chaque bâtiment
    (composante connexe de toit/mur) est un objet entier, le reste est découpé
    en superpixels d'environ 2,5 m."""

    def __init__(self, img, lab, ppm, prof, instances=None):
        self.lab = lab
        self.prof = prof
        self.instances = instances     # carte des bâtiments si le dataset les individualise
        cls = np.full(lab.shape, core.URBAN, np.uint8)
        cls[np.isin(lab, prof.grass)] = core.RANGELAND
        cls[np.isin(lab, prof.veg) & ~np.isin(lab, prof.grass)] = core.FOREST
        cls[np.isin(lab, prof.bare)] = core.BARREN
        cls[np.isin(lab, prof.water)] = core.WATER
        super().__init__(img, cls, has_mask=True, ppm=ppm)
        self.min_roof_objects = 1
        self.fine_detail = True
        # images post-événement (comme RescueNet) : pas de flammes ni de fumée
        self.allow_active_fire = False
        self.epicenter_by_area = True

    def _segment(self):
        lab, prof = self.lab, self.prof
        if self.instances is not None:
            # le dataset donne déjà chaque bâtiment : on garde son découpage
            bl = self.instances.astype(np.int32)
        else:
            building = np.isin(lab, prof.building)
            bmask = cv2.morphologyEx(building.astype(np.uint8), cv2.MORPH_OPEN, core.disk(self.m(0.3)))
            # Des bâtiments collés (toits ou murs qui se touchent) comptent pour un
            # seul bâtiment : composantes connexes de toit + mur, en tolérant les
            # fentes d'annotation étroites (voir TOUCH_GAP_M).
            closed = cv2.morphologyEx(bmask, cv2.MORPH_CLOSE, core.disk(self.m(TOUCH_GAP_M)))
            _, comp = cv2.connectedComponents(closed, connectivity=8)
            bl = np.where(bmask > 0, comp, 0).astype(np.int32)
            # éclats d'annotation (petits et sans toit) : ce ne sont pas des bâtiments
            nb = int(bl.max()) + 1
            area = np.bincount(bl.ravel(), minlength=nb) / self.ppm ** 2
            roof_area = np.bincount(bl.ravel(), (lab == prof.C["roof"]).ravel().astype(np.float64), nb) / self.ppm ** 2
            keep = (area >= FRAGMENT_M2) | (roof_area >= FRAGMENT_ROOF_M2)
            keep[0] = False
            bl = (np.cumsum(keep) * keep)[bl].astype(np.int32)
        bmask = (bl > 0).astype(np.uint8)
        cell = max(4.0, self.m(2.5))
        n = max(30, int(self.h * self.w / cell ** 2))
        sp = slic(self.img, n_segments=n, compactness=10, sigma=0.5,
                  start_label=0, channel_axis=-1).astype(np.int32)
        seg = sp.copy()
        seg[bl > 0] = int(sp.max()) + 1 + bl[bl > 0] - 1
        _, seg = np.unique(seg, return_inverse=True)
        self.seg = seg.reshape(lab.shape).astype(np.int32)
        self.nseg = int(self.seg.max()) + 1
        self.area = np.maximum(np.bincount(self.seg.ravel(), minlength=self.nseg), 1).astype(np.float64)

        veg_px = np.isin(lab, prof.veg)
        water_px = np.isin(lab, prof.water)
        roof_px = bmask.astype(bool)
        self.building_labels = bl
        self.veg_s = self.segmean(veg_px.astype(np.float32)) > 0.5
        self.water_s = self.segmean(water_px.astype(np.float32)) > 0.5
        self.roof_s = self.segmean(roof_px.astype(np.float32)) > 0.5
        self.veg, self.water, self.roof = veg_px, water_px, roof_px
        self.ground = ~self.veg & ~self.roof & ~self.water
        road = np.isin(lab, prof.road)
        # la chaussée ne brûle pas ; chaussée et piscines ne sont pas décapées
        self.burnable = ~road
        self.soil = self.ground & ~road & ~np.isin(lab, prof.pool)
        self.ignite_on_roofs = True
        self.road_px = road
        self.cent_x = self.segmean(self.xx)
        self.cent_y = self.segmean(self.yy)


# ---------------------------------------------------------------------------
# Inondation réaliste (vue de drone)
# ---------------------------------------------------------------------------
# Modèle "baignoire avec connectivité" (bathtub + connectivity) : la surface de
# l'eau est PLATE (un seul niveau), et n'est inondé que ce qui est plus bas que
# ce niveau ET relié à une source (eau existante ou bord bas de l'image). Une
# cuvette isolée reste sèche. Les rues, plus basses que trottoirs et jardins,
# sont envahies en premier ; les maisons sont des obstacles (l'eau les entoure).
def _flood_extent(H, passable, seeds, level):
    """Zone inondée au niveau `level` : H < level, praticable, reliée aux germes."""
    wet = ((H < level) & passable).astype(np.uint8)
    n, lab = cv2.connectedComponents(wet, connectivity=4)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[seeds & (wet > 0)])] = True
    keep[0] = False
    return keep[lab]


def _streaks(sc, rng, angle, length_m):
    """Bruit étiré dans le sens du courant (traînées de turbidité)."""
    base = sc.noise(rng, 1.5, 3)
    L = int(max(3, sc.m(length_m))) | 1
    k = np.zeros((L, L), np.float32)
    c = L // 2
    dx, dy = np.cos(angle) * c, np.sin(angle) * c
    cv2.line(k, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, 1)
    k /= k.sum()
    return core.normalize(cv2.filter2D(base, -1, k))


def simulate_flood_drone(sc, rng, severity=0.35):
    h, w = sc.h, sc.w
    f = sc.f
    lab = sc.lab
    building = sc.roof
    water0 = sc.water
    road = sc.road_px

    # --- terrain fictif (mètres)
    # Avec de l'eau visible (rivière, canal, lac) : le sol monte en s'éloignant
    # de la rive -> la crue déborde de l'eau existante, qui est la seule source.
    # Sans eau : pente générale aléatoire, l'eau arrive par le bord le plus bas.
    has_water = water0.mean() > 0.005
    if has_water:
        dist_m = cv2.distanceTransform((~water0).astype(np.uint8), cv2.DIST_L2, 5) / sc.ppm
        H = 0.03 * dist_m                                # +3 m tous les 100 m
        # sens du courant : le long de la rive (orientation principale de l'eau)
        ys, xs = np.nonzero(water0[::4, ::4])
        cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
        ev = np.linalg.eigh(cov)[1][:, -1]
        th = float(np.arctan2(ev[1], ev[0]))
    else:
        th = rng.uniform(0, 2 * np.pi)                  # direction de la pente (vers le bas)
        slope = rng.uniform(0.008, 0.02)                # 0,8 à 2 m pour 100 m
        proj = (sc.xx * np.cos(th) + sc.yy * np.sin(th)) / sc.ppm
        H = -slope * proj
    H = H + 0.35 * (sc.noise(rng, 60, 3) - 0.5) + 0.08 * (sc.noise(rng, 6, 2) - 0.5)
    H = H.astype(np.float32)
    # les rues (et leurs fossés) sont les points bas du quartier : sur les
    # images réelles (FloodNet / Harvey) l'eau suit d'abord le réseau de rues
    road_d = cv2.distanceTransform((~road).astype(np.uint8), cv2.DIST_L2, 5) / sc.ppm
    H -= 0.30 * np.exp(-road_d / 3.0)                  # pente douce vers la rue
    H[road] -= 0.10                                    # chaussée sous le trottoir
    H[sc.veg | sc.soil] += 0.05
    H[water0] -= 1.0                                   # plans d'eau existants

    # --- sources : eau existante + bord de l'image situé le plus bas
    border = np.zeros((h, w), bool)
    b = max(2, int(0.01 * min(h, w)))
    border[:b, :] = border[-b:, :] = True
    border[:, :b] = border[:, -b:] = True
    low_edge = border & (H <= np.quantile(H[border], 0.25))
    seeds = water0 if has_water else low_edge
    passable = ~building

    # --- niveau d'eau : recherche dichotomique pour inonder `severity` de la surface praticable
    # la sévérité porte sur la terre ferme : l'eau déjà présente ne compte pas
    target = severity * (passable & ~water0).sum() + water0.sum()
    lo, hi = float(H.min()), float(np.quantile(H[passable], 0.995)) + 0.5
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if _flood_extent(H, passable, seeds, mid).sum() < target:
            lo = mid
        else:
            hi = mid
    level = hi
    flooded = _flood_extent(H, passable, seeds, level) | water0
    depth = np.clip(level - H, 0, None) * flooded

    # --- couleur de l'eau, d'après les images réelles FloodNet (Harvey) :
    #  - eau de crue en quartier : brun "thé" foncé, presque noire en profondeur ;
    #  - grande étendue / rivière en crue : boue brun clair uniforme
    #    (mesuré sur FloodNet : RGB ~ 114, 102, 85, très peu texturée) ;
    #  - eau peu profonde : on voit le sol à travers, teinté ambre (sédiments) ;
    #  - reflets du ciel gris-bleu, en grandes plages douces ;
    #  - surface lisse : pas de texture héritée du sol sous l'eau.
    dark_tea = np.array([62, 50, 38], np.float32)
    muddy = np.array([114, 102, 85], np.float32)
    turb = float(rng.uniform(0, 1)) if not has_water else float(rng.uniform(0.5, 1))
    deep_col = dark_tea * (1 - turb) + muddy * turb
    deep_col = deep_col * rng.uniform(0.9, 1.08)
    flow = th
    st = _streaks(sc, rng, flow, 15)[..., None]
    col = deep_col * (0.95 + 0.1 * sc.noise(rng, 40, 2))[..., None] * (0.97 + 0.06 * st)
    d = np.clip(depth / 1.2, 0, 1)[..., None]
    col = col * (1.08 - 0.16 * d)                                      # plus sombre au fond
    # reflet du ciel (gris-bleu) en grandes plages
    sky = np.array([150, 160, 172], np.float32)
    sheen = np.clip((sc.noise(rng, 25, 3) - 0.45) * 1.6, 0, 1) * rng.uniform(0.15, 0.35)
    col = core.blend(col, sky, sheen.astype(np.float32))
    # ombres d'origine (maisons, arbres) : elles ASSOMBRISSENT l'eau, mais la
    # texture du sol immergé ne doit pas transparaître (eau turbide)
    light = cv2.GaussianBlur(sc.lum, (0, 0), max(1.0, sc.m(1.0)))
    ref = np.median(light[flooded]) if flooded.any() else light.mean()
    shade = np.clip(light / (ref + 1e-6), 0.4, 1.0)
    shade = np.where(shade > 0.8, 1.0, shade / 0.8 * 0.95)              # seulement les ombres
    col = col * shade[..., None]
    # eau peu profonde (< ~15 cm) : sol visible, teinté ambre, assombri
    amber = f * np.array([0.72, 0.6, 0.45], np.float32) + np.array([18, 10, 0], np.float32)
    shallow = core.smoothstep(depth / 0.15)
    col = col * shallow[..., None] + amber * (1 - shallow[..., None])

    # --- opacité : bord net (pas d'auréole), eau quasi opaque dès 15 cm
    a = (0.75 + 0.25 * core.smoothstep(depth / 0.15)) * (depth > 0)
    # arbres : houppiers au-dessus de l'eau ; pelouse / haie basse : noyées
    k = max(1.0, sc.m(0.5))
    mu = cv2.GaussianBlur(sc.lum, (0, 0), k)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(sc.lum ** 2, (0, 0), k) - mu ** 2, 0))
    canopy = sc.veg & (sd > np.percentile(sd[sc.veg], 55) if sc.veg.any() else False)
    canopy = cv2.morphologyEx(canopy.astype(np.uint8), cv2.MORPH_OPEN, core.disk(sc.m(0.4))).astype(bool)
    a = np.where(canopy, a * 0.15, a)
    # véhicules : partiellement puis totalement immergés
    car = lab == sc.prof.car
    a = np.where(car, np.clip((depth - 0.3) / 1.0, 0, 0.9), a)
    a = a * flooded
    out = core.blend(f, col, a)

    # --- sol détrempé : bande étroite (~40 cm) juste au-delà de l'eau, sans
    # auréole floue (sur les vraies images le bord de l'eau est net)
    ring = cv2.dilate(flooded.astype(np.uint8), core.disk(max(1, sc.m(0.4)))).astype(bool) & ~flooded & ~building
    wet = ring * (0.5 + 0.5 * sc.noise(rng, 0.8, 2)) * 0.18
    out = out * (1 - wet)[..., None]

    # --- maisons endommagées par l'eau
    # On regarde une bande de ~0,5 m le long des murs (le pied de façade) :
    #  - "contact" = l'eau touche vraiment le mur : pixel inondé avec au moins
    #    5 cm d'eau (en dessous : sol juste mouillé / flaque, pas d'entrée d'eau) ;
    #  - longueur de contact en mètres = pixels de contact / largeur de bande ;
    #  - hauteur d'eau contre le mur = 90e centile des profondeurs de contact.
    # Une maison est ENDOMMAGÉE si l'eau la touche directement ET :
    #  (a) hauteur d'eau >= 0,15 m sur au moins 1 m de mur : l'eau dépasse le
    #      seuil des portes / bouches d'aération et entre dans le bâtiment ;
    #  (b) ou contact sur >= 3 m de mur, quelle que soit la hauteur : une façade
    #      entière les pieds dans l'eau (fondations, murs et sols imbibés) ;
    #  (c) ou contact sur >= 25 % du pourtour : maison en grande partie cernée.
    # Un simple effleurement (un angle, < 1 m, quelques cm) ne compte pas.
    bl = sc.building_labels
    nb = int(bl.max()) + 1
    flooded_bld = np.zeros(nb, bool)
    if nb > 1:
        band_px = max(2.0, sc.m(0.5))
        dil = cv2.dilate(bl.astype(np.float32), core.disk(band_px))
        rim = (dil > 0) & (bl == 0) & ~water0
        rl = dil.astype(np.int32)[rim]
        contact = flooded[rim] & (depth[rim] >= 0.05)
        perim_px = np.bincount(rl, minlength=nb).astype(np.float64)
        cont_px = np.bincount(rl, weights=contact.astype(np.float64), minlength=nb)
        contact_m = cont_px / band_px / sc.ppm
        share = cont_px / np.maximum(perim_px, 1)
        hwall = np.zeros(nb)
        dd = depth[rim][contact]
        ll = rl[contact]
        for i in np.unique(ll):
            hwall[i] = np.percentile(dd[ll == i], 90)
        flooded_bld = (((hwall >= 0.15) & (contact_m >= 1.0))
                       | (contact_m >= 3.0)
                       | ((share >= 0.25) & (contact_m >= 1.0)))
        flooded_bld[0] = False

    # --- laisse de débris : bord de l'eau + contre les maisons inondées
    edge = flooded & ~cv2.erode(flooded.astype(np.uint8), core.disk(sc.m(0.6))).astype(bool)
    against = flooded & cv2.dilate((flooded_bld[bl] & (bl > 0)).astype(np.uint8),
                                   core.disk(sc.m(1.2))).astype(bool)
    prob = np.zeros((h, w), np.float32)
    prob[edge] = 0.35
    prob[against] = np.maximum(prob[against], 0.25)
    if prob.any():
        p, v, c = core.chips(sc, rng, edge | against, building | sc.veg, f, cell_m=0.25, prob=prob)
        p = p * 0.8 + col * 0.2                                       # débris mouillés, salis
        out = core.draw_pieces(sc, out, p, v, c, shadow_m=0.05, crack_dark=0.8)
        out = core.blend(out, np.array([196, 188, 166], np.float32),
                         core.soft(edge & (sc.noise(rng, 0.3, 2) > 0.6), 0.8) * 0.5)  # écume

    damage = (flooded & ~water0) | (flooded_bld[bl] & (bl > 0))
    alpha = np.maximum(a * 0.3, wet)       # eau lisse : peu de grain capteur
    return sc.finish(out, alpha, rng), damage


# l'inondation des images de drone remplace celle du moteur satellite
core.SIMULATORS["flood"] = simulate_flood_drone


# ---------------------------------------------------------------------------
# Séisme : même rendu "gravats" que les villas, pour TOUS les bâtiments
# ---------------------------------------------------------------------------
# Rendu par fragmentation (lit de gravats, morceaux brisés déplacés/tournés,
# grandes dalles, éclats projetés, planches, poussière), avec des tailles de
# morceaux proportionnelles à la taille du bâtiment (facteur k) : une villa et
# un entrepôt s'effondrent de la même façon, à leur échelle.
# Effondrement partiel : limite IRRÉGULIÈRE (bruit) entre partie à terre et
# partie debout, et bande de transition endommagée (tuiles/tôles déplacées,
# trous) : pas de coupure nette.
DUST = np.array([186, 178, 162], np.float32)


def _fallen_sheets(sc, rng, out, sub, f, U, k):
    """Pans de toiture retombés : grands morceaux irréguliers de la matière
    d'origine, légèrement déplacés et tournés, avec une luminosité propre à
    chaque morceau et des gravats visibles entre eux. Aucun ombrage périodique :
    il créait des bandes régulières, fausses sur les toits-terrasses."""
    region = sub & (sc.noise(rng, 6 * k, 3) > 0.42)
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN,
                              core.disk(max(1, sc.m(1.0)))).astype(bool)
    if not region.any():
        return out, region
    src = (f * 0.9 + DUST * 0.1).astype(np.float32)
    p, v, c, _ = core.shatter(sc, rng, region, src, cell_m=2.5 * k, move_m=0.25 * k,
                              rot_deg=6, drop=0.3, tilt=0.12)
    v = v & sub
    o = sc.m(0.35)
    shd = cv2.GaussianBlur(core.shift(v.astype(np.float32), core.SUN[0] * o, core.SUN[1] * o),
                           (0, 0), max(1.0, sc.m(0.12)))
    out = out * (1 - 0.4 * np.clip(shd, 0, 1) * (sub & ~v))[..., None]
    out = np.where(v[..., None], p, out)
    return np.where((c & v)[..., None], out * 0.82, out), v


def _render_collapse(sc, rng, out, foot, sub, trans, intact, k):
    """foot : bâtiments touchés ; sub : parties effondrées ; trans : bandes de
    transition ; intact : bâtiments intacts (les gravats s'arrêtent contre
    leurs murs) ; k : facteur d'échelle des morceaux."""
    f = sc.f
    if not sub.any() and not trans.any():
        return out, np.zeros(foot.shape, bool)
    t = float(np.clip((k - 1) / 1.5, 0, 1))      # 0 = villa, 1 = très grand bâtiment
    dusty = (f * 0.7 + core.luminance(f)[..., None] * 0.3) * 0.9 + np.array([18, 16, 12], np.float32)
    # grands toits (tôle claire...) : morceaux plus poussiéreux et moins
    # contrastés, sinon ils ressemblent à du verre brisé
    dusty = core.blend(dusty, DUST, np.full(foot.shape, 0.08 + 0.15 * t, np.float32))
    free = ~sc.water & ~intact
    # lit de gravats : emprise effondrée + débordement irrégulier
    ring = cv2.dilate(sub.astype(np.uint8), core.disk(sc.m(1.5 * k))).astype(bool) & free
    bed = sub | (ring & (sc.noise(rng, 0.5 * k, 3) > 0.5))
    out = np.where(bed[..., None], core.rubble_base(sc, rng, f), out)
    # petits morceaux en vrac puis grandes dalles par-dessus
    # petits débris : sur les grands bâtiments, gravats poussiéreux (béton,
    # isolant) plutôt que fragments de toit, sinon effet "verre brisé"
    small_src = core.blend(dusty, DUST, np.full(foot.shape, 0.35 * t, np.float32))
    p0, v0, c0, _ = core.shatter(sc, rng, sub, small_src, cell_m=0.35 * k, move_m=0.25 * k,
                                 rot_deg=40, drop=0.92 if USE_REAL_TEX else 0.4 + 0.2 * t,
                                 tilt=0.45 - 0.3 * t)
    out = core.draw_pieces(sc, out, p0, v0, c0, shadow_m=0.06 * k,
                           crack_dark=0.88 if USE_REAL_TEX else 0.7 + 0.15 * t)
    big = sub & (sc.noise(rng, 2.5 * k, 3) > (0.62 if USE_REAL_TEX else 0.4 + 0.45 * t))
    p1, v1, c1, _ = core.shatter(sc, rng, big, dusty, cell_m=1.3 * k, move_m=0.4 * k,
                                 rot_deg=15, drop=0.45 + 0.15 * t, tilt=0.45 - 0.2 * t)
    out = core.draw_pieces(sc, out, p1, v1, c1, shadow_m=0.2 * k,
                           crack_dark=0.8 if USE_REAL_TEX else 0.55 + 0.25 * t)
    U = V = None
    sheet_reg = np.zeros(sub.shape, bool)
    if t > 0:
        # repère de chaque grand bâtiment (grand axe / petit axe) : les
        # éléments de couverture restent alignés sur le bâtiment
        _, cxb, cyb, angb = _building_frames(sc)
        bl = sc.building_labels
        U = np.zeros((sc.h, sc.w), np.float32)
        V = np.zeros((sc.h, sc.w), np.float32)
        for i in np.unique(bl[sub | trans]):
            if i == 0:
                continue
            mi = bl == i
            co, si = np.cos(angb[i]), np.sin(angb[i])
            dx, dy = sc.xx[mi] - cxb[i], sc.yy[mi] - cyb[i]
            U[mi], V[mi] = dx * co + dy * si, -dx * si + dy * co
        # grands bâtiments : on garde des ÉLÉMENTS DU BÂTIMENT D'ORIGINE dans
        # l'effondrement -> grands PANS DE TOITURE retombés d'un seul tenant
        # (tôles vissées entre elles : matière, couleur et nervures d'origine),
        # pliés en accordéon, aux bords déchirés, posés sur les gravats
        out, sheet_reg = _fallen_sheets(sc, rng, out, sub, f, U, k)
    # la partie restée debout projette une ombre douce sur la partie effondrée
    standing = (foot & ~sub).astype(np.float32)
    if standing.any() and sub.any():
        o = sc.m(1.5 * k)
        shd = cv2.GaussianBlur(core.shift(standing, core.SUN[0] * o, core.SUN[1] * o), (0, 0),
                               max(1.0, sc.m(0.4)))
        out = out * (1 - 0.4 * np.clip(shd, 0, 1) * sub)[..., None]
    # éclats projetés, de plus en plus rares en s'éloignant
    land = cv2.dilate(sub.astype(np.uint8), core.disk(sc.m(3 * k))).astype(bool) & ~foot & free
    d = cv2.distanceTransform((~sub).astype(np.uint8), cv2.DIST_L2, 5)
    prob = 0.55 * np.exp(-d / max(1.0, sc.m(1.0 * k)))
    p2, v2, c2 = core.chips(sc, rng, land, sub, dusty, cell_m=0.22 * k, prob=prob)
    out = core.draw_pieces(sc, out, p2, v2, c2, shadow_m=0.08 * k, crack_dark=0.75)
    out = core.scatter_planks(sc, rng, out, bed & ~sheet_reg,
                              density_m2=(0.03 if USE_REAL_TEX else 0.12) / k, max_n=int(900 * k))
    # transition des grands bâtiments : couverture affaissée en creux, avec
    # quelques tôles entières arrachées (trous rectangulaires) -> pas de
    # "confettis" de petits morceaux sur le toit
    if trans.any() and U is not None:
        nz = lambda m_, oc=2: sc.noise(rng, m_, oc)
        sag = _sagged_roof(sc, rng, f, U, V, trans, nz, depth=0.35)
        out = np.where(trans[..., None], sag, out)

    # transition : tuiles/tôles déplacées et trous laissant voir l'intérieur
    elif trans.any():
        p3, v3, c3, _ = core.shatter(sc, rng, trans, f, cell_m=0.45 * k, move_m=0.1 * k,
                                     rot_deg=8, drop=0.35, tilt=0.25)
        holes = trans & ~v3
        attic = np.array([46, 40, 36], np.float32) * (0.7 + 0.6 * sc.noise(rng, 0.3, 2))[..., None]
        attic = attic * 0.65 + core.rubble_base(sc, rng, f) * 0.35       # gravats au fond
        out = np.where(holes[..., None], attic, out)
        out = core.draw_pieces(sc, out, p3, v3 & trans, c3 & trans, shadow_m=0.06 * k,
                               crack_dark=0.75)
    # voile de poussière
    dust = core.soft(bed, sc.m(3 * k)) * 0.12 * sc.noise(rng, 2 * k, 2)
    out = core.blend(out, DUST, dust)
    return out, bed | v2 | trans


# ---------------------------------------------------------------------------
# Dégâts PARTIELS : une partie du bâtiment détruite, le reste intact, avec
# une transition naturelle (d'après des vues aériennes réelles : Antakya,
# séisme de 2023 ; relevés de dégâts de tornade)
# ---------------------------------------------------------------------------
#  séisme : toit intact -> bord de toiture cassé (tuiles/tôles déplacées,
#           trous de plus en plus nombreux) -> structure éventrée (murs,
#           planchers) au droit de la rupture -> pans de toiture retombés,
#           inclinés, encore reconnaissables -> amas de gravats mêlant les
#           matériaux du bâtiment, qui déborde ; ombre de la partie debout.
#  tornade: couverture intacte -> bordure de tuiles/tôles arrachées par
#           endroits -> voligeage (panneaux de bois) à nu -> chevrons à nu
#           côté vent ; débris de couverture projetés sous le vent.
def _bframe(sc, i, frames):
    area, cx, cy, ang = frames
    co, si = np.cos(ang[i]), np.sin(ang[i])
    dx, dy = sc.xx - cx[i], sc.yy - cy[i]
    return dx * co + dy * si, -dx * si + dy * co, float(np.sqrt(area[i]))


def _split(sc, rng, fi, cxi, cyi, size_m, frac, th):
    """Distance signée (m) à une ligne de rupture irrégulière : > 0 côté
    détruit, < 0 côté intact ; la partie détruite couvre `frac` du bâtiment."""
    p = (sc.xx - cxi) * np.cos(th) + (sc.yy - cyi) * np.sin(th)
    p = p + (sc.noise(rng, max(2.0, 0.3 * size_m), 3) - 0.5) * 0.35 * size_m * sc.ppm
    p = p + (sc.noise(rng, max(0.8, 0.06 * size_m), 2) - 0.5) * 0.08 * size_m * sc.ppm
    thr = np.quantile(p[fi], 1 - frac)
    return (p - thr) / sc.ppm


def _interior(sc, rng, U, V):
    """Intérieur éventré vu de dessus : sols poussiéreux sombres, gravats, et
    têtes de murs intérieurs (cloisons tous les ~3-4 m)."""
    floor = core.rubble_base(sc, rng, sc.f) * 0.5 + np.array([12, 11, 10], np.float32)
    t = max(1.0, sc.m(0.22))
    walls = (np.mod(U, sc.m(3.6)) < t) | (np.mod(V, sc.m(3.1)) < t)
    walls &= sc.noise(rng, 2.0, 2) > 0.3                       # murs en partie tombés
    wall = np.array([176, 170, 160], np.float32) * (0.8 + 0.3 * sc.noise(rng, 0.5, 2))[..., None]
    tex = _scene_rubble(sc, rng, "toit") if _ACTIVE_TEX == "tornade" else False
    if tex is not False:
        return tex.copy()          # tornade : charpente / voliges à nu
    return np.where(walls[..., None], wall, floor)


def _quake_partial(sc, rng, out, fi, i, frames, intact, frac, th=None):
    f = sc.f
    U, V, size = _bframe(sc, i, frames)
    _, cxs, cys, _ = frames
    if th is None:
        th = rng.uniform(0, 2 * np.pi)
    d = _split(sc, rng, fi, cxs[i], cys[i], size, frac, th)
    C = fi & (d > 0)
    if not C.any():
        return out, np.zeros(fi.shape, bool)
    k = float(np.clip(size / 15.0, 1.0, 2.5))
    dmax = float(d[C].max())
    free = ~sc.water & ~intact

    # --- 1. bord de toiture cassé, côté intact
    wt = float(np.clip(0.12 * size, 1.5, 5.0))
    T = fi & (d <= 0) & (d > -wt)
    prob = np.clip(1 + d / wt, 0, 1)                            # 1 à la rupture, 0 à -wt
    hole = T & (sc.noise(rng, 0.7 * k, 3) < 0.1 + 0.8 * prob ** 1.4)
    hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_OPEN,
                            core.disk(max(1, sc.m(0.15)))).astype(bool)
    out = np.where(hole[..., None], _interior(sc, rng, U, V), out)
    disp = T & ~hole & (sc.noise(rng, 0.5 * k, 2) < 0.2 + 0.7 * prob)
    p3, v3, c3, _ = core.shatter(sc, rng, disp, f, cell_m=0.4 * k, move_m=0.08 * k,
                                 rot_deg=7, drop=0.0, tilt=0.2)
    out = core.draw_pieces(sc, out, p3, v3 & disp, c3 & disp, shadow_m=0.05 * k, crack_dark=0.8)

    # --- 2. côté effondré : lit de gravats + débordement
    ring = cv2.dilate(C.astype(np.uint8), core.disk(sc.m(1.5 * k))).astype(bool) & free & ~fi
    bed = C | (ring & (sc.noise(rng, 0.5 * k, 3) > 0.5))
    out = np.where(bed[..., None], core.rubble_base(sc, rng, f), out)
    roof_tex = _scene_rubble(sc, rng, "toit") if _ACTIVE_TEX == "tornade" else False
    if roof_tex is not False:
        out = np.where(C[..., None], roof_tex, out)              # tornade : toit arraché, charpente à nu
    mix = (f * 0.75 + DUST * 0.25).astype(np.float32)            # matériaux du bâtiment, poussiéreux
    p0, v0, c0, _ = core.shatter(sc, rng, C, mix, cell_m=0.3 * k, move_m=0.25 * k,
                                 rot_deg=40, drop=0.92 if USE_REAL_TEX else 0.35, tilt=0.35)
    out = core.draw_pieces(sc, out, p0, v0, c0, shadow_m=0.05 * k,
                           crack_dark=0.88 if USE_REAL_TEX else 0.75)
    # pans de toiture retombés près de la rupture : grands morceaux encore
    # reconnaissables (texture d'origine), plus sombres en s'éloignant (pente)
    if size >= np.sqrt(LARGE_M2):
        # grand bâtiment : pans de couverture CONTINUS retombés (tôles vissées
        # entre elles), pliés en accordéon -> pas d'éclats façon verre brisé
        out, v1 = _fallen_sheets(sc, rng, out, C & (d < 0.75 * dmax), f, U, k)
    else:
        S = C & (d < 0.65 * dmax) & (sc.noise(rng, 3 * k, 3) > 0.35)
        slab_src = (f * 0.92 + DUST * 0.08) * (1 - 0.22 * np.clip(d / max(dmax, 1e-3), 0, 1))[..., None]
        cell = float(np.clip(0.12 * size, 1.2, 5.0))
        p1, v1, c1, _ = core.shatter(sc, rng, S, slab_src.astype(np.float32), cell_m=cell,
                                     move_m=0.2 * k, rot_deg=8, drop=0.2, tilt=0.3)
        out = core.draw_pieces(sc, out, p1, v1, c1, shadow_m=0.25 * k, crack_dark=0.75)
    # éclats et planches au pied, côté effondré
    land = cv2.dilate(C.astype(np.uint8), core.disk(sc.m(3 * k))).astype(bool) & ~fi & free
    dist = cv2.distanceTransform((~C).astype(np.uint8), cv2.DIST_L2, 5)
    p2, v2, c2 = core.chips(sc, rng, land, C, mix, cell_m=0.22 * k,
                            prob=0.5 * np.exp(-dist / max(1.0, sc.m(1.0 * k))))
    out = core.draw_pieces(sc, out, p2, v2, c2, shadow_m=0.06 * k, crack_dark=0.8)
    out = core.scatter_planks(sc, rng, out, bed & ~v1, density_m2=(0.03 if USE_REAL_TEX else 0.1) / k,
                              max_n=int(600 * k))

    # --- 3. ombre de la partie restée debout sur les gravats + poussière
    stand = (fi & ~C).astype(np.float32)
    o = sc.m(2.0 * k)
    shd = cv2.GaussianBlur(core.shift(stand, core.SUN[0] * o, core.SUN[1] * o), (0, 0),
                           max(1.0, sc.m(0.4)))
    out = out * (1 - 0.35 * np.clip(shd, 0, 1) * bed)[..., None]
    dust = core.soft(bed, sc.m(2.5 * k)) * 0.12 * sc.noise(rng, 2 * k, 2)
    out = core.blend(out, DUST, dust)
    return out, bed | T | v2


def _tornado_partial(sc, rng, out, fi, i, frames, frac, wind):
    f = sc.f
    U, V, size = _bframe(sc, i, frames)
    _, cxs, cys, _ = frames
    # côté au vent = côté arraché
    d = _split(sc, rng, fi, cxs[i], cys[i], size, frac, wind + np.pi)
    R = fi & (d > 0)
    if not R.any():
        return out, np.zeros(fi.shape, bool)
    k = float(np.clip(size / 15.0, 1.0, 2.5))
    dmax = float(d[R].max())
    # voligeage (panneaux de bois 1,2 x 2,4 m) à nu
    # panneaux posés en quinconce, chacun avec sa propre teinte (bois neuf,
    # grisé, humide), joints discrets
    row = np.floor(V / sc.m(1.2)).astype(np.int64)
    col = np.floor((U + (row % 2) * sc.m(1.2)) / sc.m(2.4)).astype(np.int64)
    pid = (row * 7919 + col * 104729) % 100003
    tone = rng.uniform(0.82, 1.1, 100003).astype(np.float32)[pid]
    t = max(1.0, sc.m(0.04))
    seams = (np.mod(V, sc.m(1.2)) < t) | (np.mod(U + (row % 2) * sc.m(1.2), sc.m(2.4)) < t)
    deck = np.array([172, 146, 108], np.float32) * tone[..., None]
    deck = deck * (0.9 + 0.2 * sc.noise(rng, 1.5, 3))[..., None]
    deck = deck * (0.95 + 0.1 * sc.noise(rng, 0.12, 2))[..., None]
    deck = np.where(seams[..., None], deck * 0.88, deck)
    deck = core.blend(deck, f * 0.6 + 40, np.full(fi.shape, 0.15, np.float32))   # salissures
    out = np.where(R[..., None], deck, out)
    # chevrons à nu sur la partie la plus exposée, au bord au vent
    raf = R & (d > 0.55 * dmax) & (sc.noise(rng, 2 * k, 3) > 0.4)
    if raf.any():
        out = np.where(raf[..., None], core.rafters(sc, rng, raf), out)
    # bordure : tuiles / tôles arrachées par endroits, soulevées, de part et
    # d'autre de la limite, qui s'estompent vers la partie intacte
    band = fi & (d > -1.5) & (d < 0.8)
    pr = np.clip((d + 1.5) / 2.3, 0, 1)
    p, v, c, _ = core.shatter(sc, rng, band, f, cell_m=0.35, move_m=0.12, rot_deg=12,
                              drop=0.0, tilt=0.35)
    keep = band & (sc.noise(rng, 0.5, 2) > pr)
    exposed = band & ~keep & (d <= 0)
    out = np.where(exposed[..., None], deck, out)
    out = core.draw_pieces(sc, out, p, v & keep, c & keep, shadow_m=0.06, crack_dark=0.8)
    # débris de couverture projetés sous le vent
    bl = sc.building_labels
    down = ((sc.xx - cxs[i]) * np.cos(wind) + (sc.yy - cys[i]) * np.sin(wind)) > 0
    near = cv2.dilate(fi.astype(np.uint8), core.disk(sc.m(20))).astype(bool)
    land = near & down & ~(bl > 0) & ~sc.water
    dist = cv2.distanceTransform((~fi).astype(np.uint8), cv2.DIST_L2, 5)
    p2, v2, c2 = core.chips(sc, rng, land, R, f, cell_m=0.3,
                            prob=0.035 * np.exp(-dist / max(1.0, sc.m(5))))
    out = core.draw_pieces(sc, out, p2, v2, c2, shadow_m=0.05, crack_dark=0.92)
    return out, R | band | v2


TORNADO_WIDE_PROB = 0.8    # part des photos prises en plein dans le passage de la tornade


def _tornado_track(sc, rng, severity, wide=False):
    """Couloir de tornade : trajectoire rectiligne légèrement sinueuse qui
    traverse l'image en passant par une zone bâtie, largeur variable,
    intensité maximale sur l'axe et plus forte à droite de la trajectoire
    (rotation + déplacement)."""
    h, w = sc.h, sc.w
    bl = sc.building_labels
    central = np.zeros((h, w), bool)
    central[h // 5: 4 * h // 5, w // 5: 4 * w // 5] = True
    start = (bl > 0) & central
    ys, xs = np.nonzero(start if start.any() else central)
    k = rng.integers(len(ys))
    y0, x0 = float(ys[k]), float(xs[k])
    th = rng.uniform(0, np.pi)
    mx, my = np.cos(th), np.sin(th)
    nx, ny = -my, mx
    s = (sc.xx - x0) * mx + (sc.yy - y0) * my
    c = (sc.xx - x0) * nx + (sc.yy - y0) * ny
    D = float(np.hypot(h, w))
    ss = np.linspace(-D, D, 256)
    off = core._profile_1d(rng, 256, 5, -1, 1) * 0.06 * D           # sinuosité
    hw_m = float(np.clip(18 + 70 * severity, 20, 60))                # demi-largeur (m)
    if wide:
        hw_m = D / sc.ppm                                            # le couloir couvre toute la photo
    hw = sc.m(hw_m) * core._profile_1d(rng, 256, 7, 0.7, 1.3)       # largeur variable
    c = c - np.interp(s, ss, off).astype(np.float32)
    hwp = np.interp(s, ss, hw).astype(np.float32)
    u = np.abs(c) / hwp
    I = core.smoothstep(1.1 - u) * (1 + 0.25 * np.tanh(c / hwp))
    I = I * (0.75 + 0.5 * sc.noise(rng, 15, 2))
    I = np.clip(I, 0, 1).astype(np.float32)
    I[sc.water] = 0
    # vent au sol : déplacement + composante vers l'axe (motif convergent)
    vx = mx - np.sign(c) * nx * 0.9
    vy = my - np.sign(c) * ny * 0.9
    vn = np.hypot(vx, vy) + 1e-6
    return I, c, (vx / vn).astype(np.float32), (vy / vn).astype(np.float32), (nx, ny)


def _with_tornado_tex(simulate):
    """Active les textures de tornade le temps de la simulation, si demandé."""
    def wrapped(*args, **kwargs):
        global _ACTIVE_TEX
        if not USE_TORNADO_TEX:
            return simulate(*args, **kwargs)
        _ACTIVE_TEX = "tornade"
        try:
            return simulate(*args, **kwargs)
        finally:
            _ACTIVE_TEX = "gravats"
    return wrapped


@_with_tornado_tex
def simulate_tornado_drone(sc, rng, severity=0.35):
    """Tornade : même rendu des bâtiments que le séisme (effondrement total ou
    partiel), mais les bâtiments touchés sont choisis par le COULOIR de la
    trajectoire, et tout ce qui les entoure est touché aussi :
      - végétation arrachée et brunie, arbres couchés (motif convergent) ;
      - gravats réels, éclats et planches autour des bâtiments touchés, très
        denses à leur pied et de plus en plus rares en s'éloignant ;
    Hors du couloir et loin des bâtiments touchés, l'image reste intacte."""
    f = sc.f
    bl = sc.building_labels
    nb = int(bl.max()) + 1
    wide = rng.random() < TORNADO_WIDE_PROB
    I, c, vx, vy, (nx, ny) = _tornado_track(sc, rng, severity, wide)
    if nb <= 1:
        a_v = np.clip((I - 0.25) / 0.5, 0, 1) * 0.85 * sc.veg
        dead = np.array([118, 102, 76], np.float32) * (0.75 + 0.5 * sc.noise(rng, 1.5, 2))[..., None] * sc.tex
        return sc.finish(core.blend(f, dead, a_v), a_v, rng), (I > 0.3) & ~sc.water

    # --- bâtiments touchés : intensité moyenne reçue dans le couloir
    frames = _building_frames(sc)
    area = frames[0]
    cnt = np.maximum(np.bincount(bl.ravel(), minlength=nb), 1)
    Ib = np.bincount(bl.ravel(), I.ravel(), nb) / cnt
    cb = np.bincount(bl.ravel(), c.ravel(), nb) / cnt
    ids = [i for i in range(1, nb) if area[i] >= 15]
    full, part, light = [], [], []
    if wide:
        # photo dans le passage de la tornade : le hasard individuel domine, donc
        # une maison détruite peut côtoyer une maison intacte ou peu touchée ; un
        # faible effet de zone (tourbillons, ~40 m) crée quelques regroupements.
        zone = sc.noise(rng, 40, 2)
        ix = np.clip(frames[1].astype(int), 0, sc.w - 1)
        iy = np.clip(frames[2].astype(int), 0, sc.h - 1)
        score = {i: 0.2 * zone[iy[i], ix[i]] + 0.8 * rng.random() for i in ids}
        order = sorted(ids, key=lambda i: score[i])
        q_intact = rng.uniform(0.3, 0.45)
        q_light = q_intact + rng.uniform(0.15, 0.2)
        q_part = q_light + rng.uniform(0.15, 0.2)
        for r, i in enumerate(order):
            q = (r + 0.5) / len(order)
            if q < q_intact:
                continue
            if q < q_light:
                light.append(i)
            elif q < q_part or (area[i] >= LARGE_M2 and rng.random() < 0.9):
                part.append(i)
            else:
                full.append(i)
    else:
        for i in ids:
            if Ib[i] > 0.6 and not (area[i] >= LARGE_M2 and rng.random() < 0.9):
                full.append(i)                 # comme le séisme : grands surtout partiels
            elif Ib[i] > 0.35:
                part.append(i)
            elif Ib[i] > 0.2:
                light.append(i)
    hit = full + part + light
    src = np.isin(bl, hit) if hit else np.zeros((sc.h, sc.w), bool)
    intact = (bl > 0) & ~src

    # --- abords : portée irrégulière, proportionnelle au degré de dégâts
    # (détruit ~7-21 m, partiel ~4-12 m, peu touché ~2-6 m)
    halo = np.zeros((sc.h, sc.w), np.float32)
    rn = sc.noise(rng, 12, 3)
    for group, r0, r1 in ((full, 7, 14), (part, 4, 8), (light, 2, 4)):
        if group:
            d_g = cv2.distanceTransform((~np.isin(bl, group)).astype(np.uint8), cv2.DIST_L2, 5) / sc.ppm
            halo = np.maximum(halo, np.clip(1 - d_g / (r0 + r1 * rn), 0, 1).astype(np.float32))
    # dans le passage : la végétation est touchée par massifs, le sol seulement
    # autour des bâtiments touchés (le reste de la photo reste intact)
    Iv = np.maximum(0.8 * I if wide else I, 0.95 * halo)

    # --- végétation : feuillage arraché, bois mort, arbres couchés
    out = f.copy()
    a_v = np.clip((Iv - 0.25) / 0.5, 0, 1) * 0.85 * sc.veg
    if wide:
        # tous les arbres ne sont pas détruits : massifs épargnés loin des bâtiments
        survive = np.clip((sc.noise(rng, 18, 3) - 0.55) / 0.2, 0, 1)
        a_v = a_v * np.maximum(survive, np.clip(halo * 1.5, 0, 1))
    dead = np.array([118, 102, 76], np.float32) * (0.75 + 0.5 * sc.noise(rng, 1.5, 2))[..., None] * sc.tex
    out = core.blend(out, dead, a_v)
    soil = np.array([150, 132, 104], np.float32) * sc.tex
    Is = halo if wide else Iv
    out = core.blend(out, soil, np.clip((Is - 0.5) / 0.35, 0, 1) * 0.5 * getattr(sc, "soil", sc.ground))
    trunks = np.zeros((sc.h, sc.w), np.uint8)
    cand = sc.veg & (a_v > 0.25)
    if cand.any():
        ys, xs = np.nonzero(cand)
        n = min(4000, int(Iv[cand].sum() / max(1.0, sc.m(7)) ** 2))
        if n > 0:
            p = Iv[ys, xs] / Iv[ys, xs].sum()
            th_ = max(1, int(round(sc.m(0.18))))
            for j in rng.choice(len(ys), size=n, p=p):
                y, x = ys[j], xs[j]
                ang = np.arctan2(vy[y, x], vx[y, x]) + rng.normal(0, 0.25)
                L = sc.m(rng.uniform(4, 9))
                cv2.line(trunks, (int(x), int(y)), (int(x + np.cos(ang) * L), int(y + np.sin(ang) * L)),
                         255, th_, cv2.LINE_AA)
    out = core.blend(out, out * 0.55 + np.array([62, 52, 38], np.float32),
                     trunks.astype(np.float32) / 255 * 0.75)
    dmg = ((a_v > 0.25) | (halo > 0.05) | (~wide & (I > 0.3))) & ~sc.water & ~(bl > 0)

    # --- gravats et débris au sol autour des bâtiments touchés
    ground = ~(bl > 0) & ~sc.water
    if src.any():
        streak = sc.noise(rng, 2.0, 3)
        thr = 0.85 - 0.6 * halo + 0.12 * sc.veg + 0.05 * sc.road_px
        patches = ground & (halo > 0.02) & (streak > thr)
        patches = cv2.morphologyEx(patches.astype(np.uint8), cv2.MORPH_OPEN,
                                   core.disk(max(1, sc.m(0.25)))).astype(bool)
        out = core.blend(out, core.rubble_base(sc, rng, f), core.soft(patches, sc.m(0.2)) * 0.92)
        dmg |= patches
    out = out.astype(np.float32)

    # --- bâtiments : exactement le rendu du séisme
    groups = {}
    for i in full:
        k = round(float(np.clip(np.sqrt(area[i]) / 15.0, 1.0, 2.5)) * 2) / 2
        groups.setdefault(k, []).append(i)
    for k, items in sorted(groups.items()):
        foot = np.isin(bl, items)
        out, dm = _render_collapse(sc, rng, out, foot, foot.copy(), np.zeros_like(foot), intact, k)
        dmg |= dm | foot
    for i in part:
        fi = bl == i
        th = np.arctan2(-np.sign(cb[i]) * ny, -np.sign(cb[i]) * nx)   # côté de l'axe
        frac = float(rng.uniform(0.35, 0.7)) if wide else float(np.clip(0.3 + 0.75 * (Ib[i] - 0.2), 0.3, 0.65))
        out, dm = _quake_partial(sc, rng, out, fi, i, frames, intact, frac, th=th)
        dmg |= dm | fi            # bâtiment touché, même en partie : entièrement endommagé
    # moins touchés : seule une petite partie du bâtiment est détruite
    for i in light:
        fi = bl == i
        out, dm = _quake_partial(sc, rng, out, fi, i, frames, intact, float(rng.uniform(0.12, 0.25)))
        dmg |= dm | fi

    # --- éclats et planches par-dessus, projetés depuis les bâtiments touchés
    if src.any():
        p2, v2, c2 = core.chips(sc, rng, ground if wide else ground & (halo > 0), src, f, cell_m=0.35,
                                prob=0.3 * halo ** 1.5 + (0.004 if wide else 0.0))
        out = core.draw_pieces(sc, out, p2, v2, c2, shadow_m=0.06, crack_dark=0.88)
        out = core.scatter_planks(sc, rng, out, ground & (halo > 0.15), density_m2=0.05, max_n=700)
        dmg |= v2
    # un bâtiment intact reste intact : ni débris ni ombre sur son toit, sinon
    # l'image contredirait le masque
    out = np.where(intact[..., None], f, out)
    dmg &= ~intact
    alpha = np.maximum(a_v, dmg.astype(np.float32) * 0.6)
    return sc.finish(out, alpha, rng), dmg


# ---------------------------------------------------------------------------
# Tornade à intensité réglable (options --intensite / --evolution)
# ---------------------------------------------------------------------------
# Même trajectoire et mêmes tirages pour toutes les intensités d'une image
# (graine fixe) : chaque bâtiment reçoit un score d'exposition fixe, et son
# niveau de dégâts (intact < peu touché < partiel < détruit) ne peut que monter
# avec l'intensité. Chaque bâtiment est rendu avec sa propre graine, pour garder
# le même aspect tant que son niveau ne change pas.
# Les bâtiments touchés sont presque toujours les mêmes : la part touchée passe
# de INT_HIT0 (intensité faible) à INT_HIT0 + INT_HIT1 (intensité 1), les plus
# exposés d'abord. C'est la GRAVITÉ de chacun qui suit l'intensité :
#   g = x * (INT_G0 + (1 - INT_G0) * exposition relative)
# peu touché (g < INT_G_PART, petite partie abîmée), partiel (part détruite
# croissante), puis détruit (g >= INT_G_FULL).
INT_HIT0, INT_HIT1 = 0.45, 0.25
INT_G0, INT_G_PART, INT_G_FULL = 0.6, 0.3, 0.75
# niveaux de dégâts -> classes RescueNet (data/rescuenet.py du dépôt officiel)
LEVEL_COLORS = {0: (180, 120, 120),      # building-no-damage
                1: (235, 255, 7),        # building-medium-damage  (peu touché)
                2: (255, 184, 6),        # building-major-damage   (partiel)
                3: (255, 0, 0)}          # building-total-destruction (détruit)


SLAB_CONCRETE = np.array([170, 165, 156], np.float32)     # béton de dalle, un peu sali
SLAB_DEBRIS = 0.03          # probabilité de base d'un éclat sur la dalle (côté au vent)
SLAB_DEBRIS_DOWN = 0.12     # supplément côté sous le vent
SLAB_PILE = 0.20            # part de la dalle couverte par le tas de gravats sous le vent (environ)


def _render_slab(sc, rng, out, foot, i, frames, wx, wy):
    """Maison rasée par une tornade violente (d'après des vues aériennes de
    Moore 2013 ou de Greenfield 2024) : la dalle de béton apparaît presque nue,
    avec quelques bouts de cloisons au ras du sol, des débris épars, et un tas
    de gravats repoussé du côté sous le vent. Le débord du toit laisse voir le sol."""
    f = sc.f
    slab = cv2.erode(foot.astype(np.uint8), core.disk(max(1, sc.m(0.5)))).astype(bool)
    if slab.sum() < 0.3 * foot.sum():
        slab = foot.copy()
    edge = foot & ~slab
    # dalle : béton clair, nuances larges, taches d'humidité et d'arrachement
    conc = SLAB_CONCRETE * (0.86 + 0.24 * sc.noise(rng, 1.5, 3))[..., None]
    conc = conc * (0.93 + 0.1 * sc.noise(rng, 0.25, 2))[..., None]
    stain = sc.noise(rng, 0.9, 2) > 0.7
    conc = np.where(stain[..., None], conc * 0.72 + np.array([18, 15, 10], np.float32), conc)
    out = np.where(slab[..., None], conc, out)
    # sous l'ancien débord de toit : sol nu, un peu remué
    # (sol remué, mêlé de gravats, bord irrégulier : pas de cadre net autour de la dalle)
    soil = np.array([132, 118, 96], np.float32) * sc.tex * (0.75 + 0.45 * sc.noise(rng, 0.35, 3))[..., None]
    soil = core.blend(soil, core.rubble_base(sc, rng, f), (sc.noise(rng, 0.5, 2) > 0.55).astype(np.float32) * 0.8)
    out = np.where(edge[..., None], soil, out)
    # restes de cloisons au ras du sol, dans l'axe de la maison
    U, V, size = _bframe(sc, i, frames)
    t = max(1.0, sc.m(0.15))
    walls = (np.mod(U, sc.m(3.4)) < t) | (np.mod(V, sc.m(3.0)) < t)
    walls &= slab & (sc.noise(rng, 0.8, 2) > 0.72)          # quelques tronçons seulement
    sh = core.shift(walls.astype(np.float32), core.SUN[0] * sc.m(0.12), core.SUN[1] * sc.m(0.12)) > 0.5
    out = np.where((sh & ~walls & slab)[..., None], out * 0.7, out)
    out = np.where(walls[..., None], np.array([196, 190, 180], np.float32) * sc.tex, out)
    # côté sous le vent (0 au vent, 1 sous le vent)
    cx, cy = frames[1][i], frames[2][i]
    proj = ((sc.xx - cx) * wx + (sc.yy - cy) * wy) / max(1.0, sc.m(size))
    down = np.clip(proj + 0.5, 0, 1).astype(np.float32)
    # tas de gravats repoussé sous le vent
    pile = slab & (sc.noise(rng, 0.7, 3) + 0.6 * down > 1.25 - SLAB_PILE)
    pile = cv2.morphologyEx(pile.astype(np.uint8), cv2.MORPH_OPEN, core.disk(max(1, sc.m(0.2)))).astype(bool)
    out = core.blend(out, core.rubble_base(sc, rng, f), core.soft(pile, sc.m(0.15)) * 0.9)
    # débris épars sur la dalle, plus nombreux sous le vent
    mix = (f * 0.8 + DUST * 0.2).astype(np.float32)
    p, v, c = core.chips(sc, rng, foot, foot, mix, cell_m=0.3,
                         prob=SLAB_DEBRIS + SLAB_DEBRIS_DOWN * down + 0.25 * edge)
    out = core.draw_pieces(sc, out, p, v, c, shadow_m=0.05, crack_dark=0.85)
    out = core.scatter_planks(sc, rng, out, slab & (down > 0.5), density_m2=0.02, max_n=60)
    return out, foot


@_with_tornado_tex
def simulate_tornado_intensity(sc, seed, x):
    """x : intensité dans [0, 1] (0 = aucun dégât, 1 = tornade la plus forte)."""
    x = float(np.clip(x, 0, 1))
    rng = np.random.default_rng(seed)
    f = sc.f
    bl = sc.building_labels
    nb = int(bl.max()) + 1
    wide = rng.random() < TORNADO_WIDE_PROB
    I, c, vx, vy, (nx, ny) = _tornado_track(sc, rng, 0.35, wide)
    # champs aléatoires tirés une fois, indépendants de l'intensité
    zone = sc.noise(rng, 40, 2)
    rn = sc.noise(rng, 12, 3)
    survive = np.clip((sc.noise(rng, 18, 3) - 0.55) / 0.2, 0, 1)
    dead = np.array([118, 102, 76], np.float32) * (0.75 + 0.5 * sc.noise(rng, 1.5, 2))[..., None] * sc.tex
    streak = sc.noise(rng, 2.0, 3)
    r_b = rng.random(nb)
    frames = _building_frames(sc)
    area = frames[0]
    cnt = np.maximum(np.bincount(bl.ravel(), minlength=nb), 1)
    Ib = np.bincount(bl.ravel(), I.ravel(), nb) / cnt
    cb = np.bincount(bl.ravel(), c.ravel(), nb) / cnt
    ix = np.clip(frames[1].astype(int), 0, sc.w - 1)
    iy = np.clip(frames[2].astype(int), 0, sc.h - 1)
    # exposition fixe de chaque bâtiment, dans [0, 1]
    if wide:
        expo = 0.3 * zone[iy, ix] + 0.7 * r_b
    else:
        expo = np.clip(Ib / 0.8, 0, 1) * (0.8 + 0.2 * r_b)
    cand = [i for i in range(1, nb) if area[i] >= 15]
    rel = np.zeros(nb)                         # exposition relative : 1 = le plus exposé
    if wide:
        order = sorted(cand, key=lambda i: (-expo[i], i))
        for rank, i in enumerate(order):
            rel[i] = 1 - rank / max(1, len(order) - 1)
    else:
        for i in cand:
            rel[i] = expo[i] if expo[i] > 0.05 else -1      # hors du couloir : jamais touché
    share = INT_HIT0 + INT_HIT1 * x            # part de bâtiments touchés
    g = x * (INT_G0 + (1 - INT_G0) * np.clip(rel, 0, 1))
    full, part, light = [], [], []
    level = np.zeros(nb, np.int32)
    for i in cand:
        if x <= 0 or rel[i] < 1 - share:
            continue
        if g[i] >= INT_G_FULL and not (area[i] >= LARGE_M2 and g[i] < INT_G_FULL + 0.15):
            full.append(i)                     # grands bâtiments : partiels plus longtemps
            level[i] = 3
        elif g[i] >= INT_G_PART:
            part.append(i)
            level[i] = 2
        else:
            light.append(i)
            level[i] = 1
    hit = full + part + light
    src = np.isin(bl, hit) if hit else np.zeros((sc.h, sc.w), bool)
    intact = (bl > 0) & ~src

    halo = np.zeros((sc.h, sc.w), np.float32)
    reach = 0.25 + 0.75 * x                    # portée des débris au sol selon l'intensité
    for group, r0, r1 in ((full, 7, 14), (part, 4, 8), (light, 2, 4)):
        if group:
            d_g = cv2.distanceTransform((~np.isin(bl, group)).astype(np.uint8), cv2.DIST_L2, 5) / sc.ppm
            halo = np.maximum(halo, np.clip(1 - d_g / ((r0 + r1 * rn) * reach), 0, 1).astype(np.float32))
    Iv = np.maximum((0.8 * I if wide else I) * x, 0.95 * halo)

    # végétation
    out = f.copy()
    a_v = np.clip((Iv - 0.25) / 0.5, 0, 1) * 0.85 * sc.veg
    if wide:
        a_v = a_v * np.maximum(survive, np.clip(halo * 1.5, 0, 1))
    out = core.blend(out, dead, a_v)
    soil = np.array([150, 132, 104], np.float32) * sc.tex
    Is = halo if wide else Iv
    out = core.blend(out, soil, np.clip((Is - 0.5) / 0.35, 0, 1) * 0.5 * getattr(sc, "soil", sc.ground))
    trunks = np.zeros((sc.h, sc.w), np.uint8)
    cand = sc.veg & (a_v > 0.25)
    if cand.any():
        rt = np.random.default_rng(seed + 1)
        ys, xs = np.nonzero(cand)
        n = min(4000, int(Iv[cand].sum() / max(1.0, sc.m(7)) ** 2))
        if n > 0:
            p = Iv[ys, xs] / Iv[ys, xs].sum()
            th_ = max(1, int(round(sc.m(0.18))))
            for j in rt.choice(len(ys), size=n, p=p):
                y, x_ = ys[j], xs[j]
                ang = np.arctan2(vy[y, x_], vx[y, x_]) + rt.normal(0, 0.25)
                L = sc.m(rt.uniform(4, 9))
                cv2.line(trunks, (int(x_), int(y)), (int(x_ + np.cos(ang) * L), int(y + np.sin(ang) * L)),
                         255, th_, cv2.LINE_AA)
    out = core.blend(out, out * 0.55 + np.array([62, 52, 38], np.float32),
                     trunks.astype(np.float32) / 255 * 0.75)
    dmg = ((a_v > 0.25) | (halo > 0.05) | (~wide & (I * x > 0.3))) & ~sc.water & ~(bl > 0)

    # gravats au sol
    ground = ~(bl > 0) & ~sc.water
    if src.any():
        thr = 0.85 - 0.6 * halo + 0.12 * sc.veg + 0.05 * sc.road_px
        patches = ground & (halo > 0.02) & (streak > thr)
        patches = cv2.morphologyEx(patches.astype(np.uint8), cv2.MORPH_OPEN,
                                   core.disk(max(1, sc.m(0.25)))).astype(bool)
        out = core.blend(out, core.rubble_base(sc, np.random.default_rng(seed + 2), f),
                         core.soft(patches, sc.m(0.2)) * 0.92)
        dmg |= patches
    out = out.astype(np.float32)

    # bâtiments : une graine par bâtiment
    for i in full:
        foot = bl == i
        k = round(float(np.clip(np.sqrt(area[i]) / 15.0, 1.0, 2.5)) * 2) / 2
        rng_i = np.random.default_rng(seed * 7919 + i)
        if area[i] < LARGE_M2:
            # maison : rasée jusqu'à la dalle, vent pris au centre du bâtiment
            out, dm = _render_slab(sc, rng_i, out, foot, i, frames,
                                   float(vx[iy[i], ix[i]]), float(vy[iy[i], ix[i]]))
        else:
            out, dm = _render_collapse(sc, rng_i, out, foot, foot.copy(), np.zeros_like(foot), intact, k)
        dmg |= dm | foot
    for i in part:
        fi = bl == i
        th = np.arctan2(-np.sign(cb[i]) * ny, -np.sign(cb[i]) * nx)
        frac = 0.25 + 0.5 * float(np.clip((g[i] - INT_G_PART) / (INT_G_FULL - INT_G_PART), 0, 1))
        out, dm = _quake_partial(sc, np.random.default_rng(seed * 7919 + i), out, fi, i, frames, intact, frac, th=th)
        dmg |= dm | fi
    for i in light:
        fi = bl == i
        frac = 0.03 + 0.2 * float(np.clip(g[i] / INT_G_PART, 0, 1))      # presque intact au début
        out, dm = _quake_partial(sc, np.random.default_rng(seed * 7919 + i), out, fi, i, frames, intact, frac)
        dmg |= dm | fi

    # éclats et planches
    if src.any():
        rc = np.random.default_rng(seed + 3)
        p2, v2, c2 = core.chips(sc, rc, ground if wide else ground & (halo > 0), src, f, cell_m=0.35,
                                prob=0.3 * halo ** 1.5 + (0.004 if wide else 0.0))
        out = core.draw_pieces(sc, out, p2, v2, c2, shadow_m=0.06, crack_dark=0.88)
        out = core.scatter_planks(sc, rc, out, ground & (halo > 0.15), density_m2=0.05, max_n=700)
        dmg |= v2
    out = np.where(intact[..., None], f, out)
    dmg &= ~intact
    alpha = np.maximum(a_v, dmg.astype(np.float32) * 0.6)
    return sc.finish(out, alpha, np.random.default_rng(seed + 4)), dmg, level


def level_mask(scene, level):
    """Masque à 4 niveaux de dégâts (classes bâtiment de RescueNet) + route."""
    out = np.zeros(scene.lab.shape + (3,), np.uint8)
    out[scene.road_px] = MASK_COLORS["route"]
    bl = scene.building_labels
    if int(bl.max()) > 0:
        fp = _full_footprints(bl, scene.ppm)
        lv = np.zeros(int(bl.max()) + 1, np.int32)
        lv[:len(level)] = level[:len(lv)]
        for k, col in LEVEL_COLORS.items():
            out[(fp > 0) & (lv[fp] == k)] = col
    return out


def simulate_quake_drone(sc, rng, severity=0.4):
    """Bâtiments choisis autour d'un épicentre (proportionnellement à la
    surface) ; les plus touchés s'effondrent entièrement, les suivants en
    partie, avec une limite irrégulière et une zone de transition."""
    bl = sc.building_labels
    nb = int(bl.max()) + 1
    if nb <= 1:
        return _core_quake(sc, rng, severity)
    cnt = np.bincount(bl.ravel(), minlength=nb).astype(np.float64)
    area = cnt / sc.ppm ** 2
    ok = np.where(area >= 15)[0]
    ok = ok[ok > 0]
    if len(ok) == 0:
        return _core_quake(sc, rng, severity)
    cx = np.bincount(bl.ravel(), sc.xx.ravel(), nb) / np.maximum(cnt, 1)
    cy = np.bincount(bl.ravel(), sc.yy.ravel(), nb) / np.maximum(cnt, 1)
    epi = rng.choice(ok, p=area[ok] / area[ok].sum())
    d = np.hypot(cx - cx[epi], cy - cy[epi])
    d = d / (d[ok].max() + 1e-6)
    site = sc.noise(rng, 80, 2)
    sv = site[np.clip(cy.astype(int), 0, sc.h - 1), np.clip(cx.astype(int), 0, sc.w - 1)]
    score = 0.55 * d + 0.3 * sv + 0.15 * rng.random(nb)
    order = ok[np.argsort(score[ok])]
    cum = np.cumsum(area[order]) / area[order].sum()
    n_c = int(np.searchsorted(cum, severity) + 1)
    n_p = max(0, int(np.searchsorted(cum, min(1.0, 1.6 * severity)) + 1) - n_c)
    hit_ids = order[:n_c + n_p]
    intact = (bl > 0) & ~np.isin(bl, hit_ids)

    # regroupement par échelle de morceaux (villa ~1, grand bâtiment jusqu'à 2,5)
    kb = np.clip(np.sqrt(area) / 15.0, 1.0, 2.5)
    groups = {}
    for j, i in enumerate(hit_ids):
        groups.setdefault(round(float(kb[i]) * 2) / 2, []).append((i, j < n_c))
    out = sc.f.copy()
    dmg = np.zeros((sc.h, sc.w), bool)
    frames = _building_frames(sc)
    partial = []
    for k, items in sorted(groups.items()):
        foot = np.zeros((sc.h, sc.w), bool)
        sub = np.zeros((sc.h, sc.w), bool)
        trans = np.zeros((sc.h, sc.w), bool)
        for i, full in items:
            fi = bl == i
            # grands bâtiments : le plus souvent effondrement PARTIEL (une
            # partie en gravats, le reste intact), rendu par _quake_partial
            large = area[i] >= LARGE_M2
            if full and large and rng.random() < 0.9:
                full = False
            if not full:
                partial.append(i)
                continue
            foot |= fi
            sub |= fi
        if foot.any():
            out, dm = _render_collapse(sc, rng, out, foot, sub, trans, intact, k)
            dmg |= dm | foot
    # effondrements partiels : une partie à terre, le reste debout
    for i in partial:
        fi = bl == i
        large = area[i] >= LARGE_M2
        frac = rng.uniform(0.3, 0.6) if large else rng.uniform(0.35, 0.65)
        out, dm = _quake_partial(sc, rng, out, fi, i, frames, intact, frac)
        dmg |= dm | fi            # bâtiment touché, même en partie : entièrement endommagé
    # un bâtiment intact reste intact : ni débris ni ombre sur son toit
    out = np.where(intact[..., None], sc.f, out)
    dmg &= ~intact
    return sc.finish(out, dmg.astype(np.float32), rng), dmg


_core_quake = core.SIMULATORS["earthquake"]
core.SIMULATORS["earthquake"] = simulate_quake_drone
_core_tornado = core.SIMULATORS["tornado"]
core.SIMULATORS["tornado"] = simulate_tornado_drone


# ---------------------------------------------------------------------------
# Grands bâtiments (entrepôts, usines, bâtiments publics)
# ---------------------------------------------------------------------------
# Les simulateurs traitent un bâtiment comme un tout : parfait pour une maison,
# faux pour un entrepôt de 100 m (tout le toit en gravats ou noirci d'un bloc).
# Un grand bâtiment à charpente (acier, béton) est découpé en TRAVÉES de 6 à 8 m
# entre deux fermes ; les dégâts touchent certaines travées :
#   séisme  : effondrement partiel de quelques travées voisines -> tôles
#             affaissées et désordonnées, trous sombres sur l'intérieur
#   incendie: travées brûlées -> couverture disparue, fermes et pannes d'acier
#             apparentes au-dessus d'un intérieur carbonisé ; tôles restantes
#             rouillées / noircies ; décoloration par la chaleur à la limite
#   tornade : tôles arrachées sur quelques travées -> pannes apparentes
#             au-dessus de l'intérieur
LARGE_M2 = 500.0       # au-delà : grand bâtiment (une villa fait ~150-250 m²)
BAY_M = 7.0            # largeur d'une travée


def _building_frames(sc):
    """Pour chaque bâtiment : surface (m²), centre, orientation du grand axe."""
    bl = sc.building_labels
    nb = int(bl.max()) + 1
    lab = bl.ravel()
    m = (lab > 0).astype(np.float64)
    cnt = np.bincount(lab, minlength=nb).astype(np.float64)
    x, y = sc.xx.ravel().astype(np.float64), sc.yy.ravel().astype(np.float64)
    cx = np.bincount(lab, x, nb) / np.maximum(cnt, 1)
    cy = np.bincount(lab, y, nb) / np.maximum(cnt, 1)
    dx, dy = x - cx[lab], y - cy[lab]
    mu20 = np.bincount(lab, dx * dx * m, nb) / np.maximum(cnt, 1)
    mu02 = np.bincount(lab, dy * dy * m, nb) / np.maximum(cnt, 1)
    mu11 = np.bincount(lab, dx * dy * m, nb) / np.maximum(cnt, 1)
    ang = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    # orientation des MURS (rectangle englobant minimal) plutôt que l'axe
    # d'inertie, instable pour un bâtiment presque carré
    for i in range(1, nb):
        if cnt[i] / sc.ppm ** 2 < 100:
            continue
        ys, xs = np.nonzero(bl[::2, ::2] == i)
        if len(xs) < 10:
            continue
        (_, _), (rw, rh), a_deg = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        a = np.deg2rad(a_deg)
        if rh > rw:                       # grand axe = côté le plus long
            a += np.pi / 2
        ang[i] = a
    return cnt / sc.ppm ** 2, cx, cy, ang


def _frame(sc, sel, cx, cy, ang):
    """Coordonnées (u le long du grand axe, v en travers), en pixels."""
    co, si = np.cos(ang), np.sin(ang)
    u = (sc.xx - cx) * co + (sc.yy - cy) * si
    v = -(sc.xx - cx) * si + (sc.yy - cy) * co
    return u, v


def _grid(u, v, sc, step_u_m, w_u_m, step_v_m, w_v_m):
    """Lignes de charpente : fermes en travers (tous les step_u) + pannes en long."""
    su, sv = sc.m(step_u_m), sc.m(step_v_m)
    trusses = np.mod(u, su) < max(1.5, sc.m(w_u_m))
    purlins = np.mod(v, sv) < max(1.0, sc.m(w_v_m))
    return trusses, purlins


def _pick_bays(bay, nbays_ids, chosen_frac, rng, prefer=None):
    """Choisit un groupe de travées CONTIGUËS. `prefer` (poids par travée)
    oriente le choix (ex. travées réellement atteintes par le feu)."""
    ids = np.array(sorted(nbays_ids))
    n = len(ids)
    k = int(np.clip(round(chosen_frac * n), 1, n))
    if prefer is not None and prefer.sum() > 0:
        start_c = int(np.argmax(np.convolve(prefer, np.ones(k), mode="valid")))
    else:
        start_c = int(rng.integers(0, n - k + 1))
    return set(ids[start_c:start_c + k].tolist())


def _panels(sc, rng, region, src, u, v, len_m, wid_m, move_m, tilt, drop):
    """Couverture découpée en panneaux RECTANGULAIRES alignés sur le bâtiment
    (bacs acier, plaques) : chacun déplacé, ré-éclairé (inclinaison), certains
    tombés. Renvoie (panneaux, valide, joints)."""
    pu = np.floor(u / max(2.0, sc.m(len_m))).astype(np.int64)
    pv = np.floor(v / max(2.0, sc.m(wid_m))).astype(np.int64)
    pid = (pu - pu.min()) * (pv.max() - pv.min() + 1) + (pv - pv.min())
    _, pid = np.unique(pid, return_inverse=True)
    pid = pid.reshape(u.shape)
    n = int(pid.max()) + 1
    ang = rng.uniform(0, 2 * np.pi, n)
    mag = np.abs(rng.normal(0, sc.m(move_m), n))
    dx, dy = np.cos(ang) * mag, np.sin(ang) * mag
    shade = rng.uniform(1 - tilt, 1 + 0.4 * tilt, n).astype(np.float32)
    gone = rng.random(n) < drop
    h, w = u.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = (xx - dx[pid]).astype(np.float32)
    my = (yy - dy[pid]).astype(np.float32)
    pieces = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * shade[pid][..., None]
    valid = region & ~gone[pid]
    k = np.ones((3, 3), np.uint8)
    pf = pid.astype(np.float32)
    joints = (cv2.dilate(pf, k) != cv2.erode(pf, k)) & valid
    return pieces, valid, joints


def _rib_angle(orig, s):
    """Direction des nervures de la couverture (perpendiculaire au gradient dominant)."""
    g = core.luminance(orig)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)[s]
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)[s]
    th_g = 0.5 * np.arctan2(2 * (gx * gy).sum(), (gx * gx).sum() - (gy * gy).sum())
    return th_g + np.pi / 2


def _sheets(sc, x0, y0, rib, len_m=6.0, wid_m=1.0):
    """Identifiant de tôle pour chaque pixel : bandes de `wid_m` le long des
    nervures, coupées tous les `len_m`. x0, y0 : coordonnées IMAGE locales
    (même repère que l'angle des nervures). Renvoie (id, a, b) en pixels."""
    a = x0 * np.cos(rib) + y0 * np.sin(rib)          # le long des nervures
    b = -x0 * np.sin(rib) + y0 * np.cos(rib)         # en travers
    ia = np.floor(a / sc.m(len_m)).astype(np.int64)
    ib = np.floor(b / sc.m(wid_m)).astype(np.int64)
    ids = (ia - ia.min()) * (ib.max() - ib.min() + 1) + (ib - ib.min())
    _, ids = np.unique(ids, return_inverse=True)
    return ids.reshape(x0.shape), a, b


def _sheet_damage(sc, rng, cur, orig, R, sid, p_missing, interior, sheet_src, tilt, noise):
    """Tôles manquantes (p_missing par tôle, carte H x W) -> intérieur visible ;
    tôles restantes : couleur `sheet_src`, éclairage par tôle (inclinaison),
    bords des trous irréguliers, ombre des tôles sur l'intérieur."""
    n = int(sid.max()) + 1
    cnt = np.maximum(np.bincount(sid.ravel(), minlength=n), 1)
    pm = np.bincount(sid.ravel(), p_missing.ravel(), n) / cnt
    gone = rng.random(n) < pm
    shade = rng.uniform(1 - tilt, 1 + 0.4 * tilt, n).astype(np.float32)
    hole = R & gone[sid]
    # bords déchirés : grignotage irrégulier des tôles voisines d'un trou
    rag = cv2.dilate(hole.astype(np.uint8), core.disk(max(1, sc.m(0.3)))).astype(bool)
    hole = hole | (rag & R & (noise(1.0, 2) > 0.6))
    sheet = sheet_src * shade[sid][..., None]
    res = np.where(R[..., None], np.where(hole[..., None], interior, sheet), cur)
    sh = max(1, int(round(sc.m(0.35))))
    shadow = np.roll(np.roll(R & ~hole, sh, 0), sh, 1) & hole
    res = np.where(shadow[..., None], res * 0.5, res)
    edge = R & ~hole & cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return np.where(edge[..., None], res * 0.6, res), hole


def _sagged_roof(sc, rng, src, u, v, R, noise, depth=0.45):
    """Couverture affaissée : dans chaque travée, le toit plonge vers le milieu
    (creux en V doux) ; éclairage progressif des deux versants, creux plus
    sombre, et légère ondulation des tôles."""
    bay_px = sc.m(BAY_M)
    t = np.mod(u, bay_px) / bay_px * 2 - 1                  # -1 .. 1 dans la travée
    shade = (1 - 0.45 * depth * np.exp(-(t / 0.35) ** 2) + 0.2 * depth * np.sin(np.pi * t)) * (0.92 + 0.16 * noise(3, 3))
    # gondolement : petit déplacement des nervures
    h, w = u.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp = sc.m(0.25)
    mx = (xx + (noise(3, 2) - 0.5) * 2 * amp).astype(np.float32)
    my = (yy + (noise(3, 2) - 0.5) * 2 * amp).astype(np.float32)
    warped = cv2.remap(src.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped * shade[..., None]


def _quake_large(sc, rng, i, sl, s, u, v, bay, bay_ids, orig, cur, cxi, cyi, severity, noise, bl):
    """Séisme sur un grand bâtiment à charpente (d'après les relevés post-séisme
    de bâtiments industriels : effondrement de la TRAVÉE D'EXTRÉMITÉ qui se
    propage aux travées intérieures, pannes flambées, couverture arrachée).
      - zone effondrée C : les tôles de couverture (entières, alignées sur les
        nervures) sont retombées dans le bâtiment : légèrement tournées,
        glissées vers le creux, éclairées différemment ; entre elles, intérieur
        rempli de gravats et pannes d'acier tordues ;
      - travée de transition T : couverture affaissée en creux, quelques tôles
        arrachées ;
      - la partie restée debout projette une ombre DOUCE sur la zone effondrée
        (différence de hauteur) ; aucun liseré noir ;
      - gravats tombés au pied des murs, sur une bande étroite et irrégulière."""
    h, w = s.shape
    ids = sorted(bay_ids)
    n = len(ids)
    k = int(np.clip(round((0.15 + 0.6 * severity + rng.normal(0, 0.06)) * n), 1, max(1, n - 1)))
    from_start = rng.random() < 0.5
    chosen = ids[:k] if from_start else ids[-k:]
    trans = [chosen[-1]] if from_start else [chosen[0]]
    coll = [b for b in chosen if b not in trans] or chosen
    if coll == chosen:
        trans = []
    C = s & np.isin(bay, coll)
    T = s & np.isin(bay, trans)
    # limite irrégulière entre effondré et debout (la rupture ne suit pas une ferme)
    C = C | (T & cv2.dilate(C.astype(np.uint8), core.disk(sc.m(2.0))).astype(bool)
             & (noise(2.0, 3) > 0.55))
    T = T & ~C

    rib = _rib_angle(orig, s)
    x_loc, y_loc = sc.xx[sl] - cxi, sc.yy[sl] - cyi

    # --- intérieur : gravats sombres + pannes d'acier tordues
    inner = core.rubble_base(sc, rng, sc.f)[sl] * 0.55 + np.array([8, 7, 6], np.float32)
    bend = (noise(3, 3) - 0.5) * sc.m(0.6)                     # flambement des pannes
    pur = np.mod(v + bend, sc.m(1.5)) < max(1.0, sc.m(0.12))
    pur &= noise(2.5, 2) > 0.35                                # pannes rompues par endroits
    steel = np.array([120, 116, 110], np.float32) * (0.75 + 0.35 * noise(0.8, 2))[..., None]
    inner = np.where(pur[..., None], steel, inner)
    res = np.where(C[..., None], inner, cur)

    # --- couverture effondrée : une NAPPE continue (les tôles restent vissées
    # entre elles), tombée dans le bâtiment, ondulée et PLIÉE en grandes
    # facettes ; quelques grandes déchirures laissent voir l'intérieur.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp = sc.m(0.5)                                          # ondulation douce des nervures
    wx = (noise(6, 3) - 0.5) * 2 * amp
    wy = (noise(6, 3) - 0.5) * 2 * amp
    src = (orig * 0.88 + core.luminance(orig)[..., None] * 0.12)        # un peu poussiéreuse
    warped = cv2.remap(src.astype(np.float32), xx + wx, yy + wy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    # pliage en ACCORDÉON le long des pannes : la nappe forme une suite de pans
    # (2 à 5 m) dont la pente alterne ; éclairage ~ pente (soleil latéral).
    # Les plis sont parallèles à la charpente, la hauteur varie aussi un peu
    # en travers (pans légèrement vrillés).
    u_m = (u - u[s].min()) / sc.ppm
    L = float(u_m[s].max()) + 1
    knots = [0.0]
    while knots[-1] < L:
        knots.append(knots[-1] + rng.uniform(3.0, 7.0))
    knots = np.array(knots)
    zk = np.cumsum(rng.choice([-1, 1], len(knots)) * rng.uniform(0.4, 1.0, len(knots)) * np.diff(np.r_[0, knots]) * 0.45)
    z = np.interp(u_m, knots, zk)
    slope = np.gradient(z, axis=1) * 0 + np.interp(u_m, knots[:-1] + np.diff(knots) / 2,
                                                   np.diff(zk) / np.diff(knots))
    twist = (noise(8, 2) - 0.5) * 0.25
    light = np.clip(1.0 + 0.8 * (slope + twist), 0.55, 1.3) * (0.95 + 0.1 * noise(3, 2))
    roof_c = warped * light[..., None]
    # déchirures : le long des nervures (les tôles se séparent à leurs
    # recouvrements), bandes allongées, ~8 % de la surface
    base = noise(3.0, 3)
    Lk = int(max(3, sc.m(5))) | 1
    ker = np.zeros((Lk, Lk), np.float32)
    c0 = Lk // 2
    ddx, ddy = np.cos(rib) * c0, np.sin(rib) * c0
    cv2.line(ker, (int(c0 - ddx), int(c0 - ddy)), (int(c0 + ddx), int(c0 + ddy)), 1.0, 1)
    streak = cv2.filter2D(base, -1, ker / ker.sum())
    thr = np.quantile(streak[C], 0.90) if C.any() else 1
    tear = C & (streak > thr)
    tear = cv2.morphologyEx(tear.astype(np.uint8), cv2.MORPH_OPEN, core.disk(max(1, sc.m(0.6)))).astype(bool)
    # brèche le long de la ligne de rupture : la nappe s'est arrachée de la
    # partie restée debout -> bande d'intérieur visible, largeur irrégulière
    stand = (s & ~C).astype(np.uint8)
    if stand.any():
        dstand = cv2.distanceTransform(1 - stand, cv2.DIST_L2, 5)
        gap_w = sc.m(0.6) + sc.m(1.6) * noise(3, 3)
        tear = tear | (C & (dstand < gap_w))
    rag = cv2.dilate(tear.astype(np.uint8), core.disk(max(1, sc.m(0.3)))).astype(bool) & C
    tear = tear | (rag & (noise(0.4, 2) > 0.55))                 # bords déchiquetés
    sheet_ok = C & ~tear
    # ombre douce de la nappe sur l'intérieur visible par les déchirures
    o = sc.m(0.4)
    sh = cv2.GaussianBlur(core.shift(sheet_ok.astype(np.float32), o * 0.7, o * 0.7), (0, 0),
                          max(0.8, sc.m(0.15)))
    res = res * (1 - 0.35 * np.clip(sh, 0, 1) * tear)[..., None]
    res = np.where(sheet_ok[..., None], roof_c, res)

    # --- travée de transition : couverture affaissée, quelques tôles arrachées
    if T.any():
        sag = _sagged_roof(sc, rng, orig, u, v, T, noise, depth=0.35)
        res = np.where(T[..., None], sag, res)

    # --- ombre douce de la partie restée debout sur la zone effondrée
    standing = (s & ~C).astype(np.float32)
    o = sc.m(1.2)
    sh = core.shift(standing, o * 0.7, o * 0.7)
    sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, sc.m(0.5)))
    res = res * (1 - 0.4 * np.clip(sh, 0, 1) * C)[..., None]

    # --- poussière légère sur l'effondrement
    dust = core.soft(C, sc.m(1.5)) * 0.1 * noise(2, 2)
    res = core.blend(res, np.array([180, 172, 158], np.float32), dust)

    # --- gravats au pied des murs, côté effondré (bande étroite irrégulière)
    Cf = np.zeros((sc.h, sc.w), bool)
    Cf[sl] = C
    dist = cv2.distanceTransform((~Cf).astype(np.uint8), cv2.DIST_L2, 5)
    reach = sc.m(1.6) * (0.4 + 0.9 * sc.noise(rng, 2.5, 3))
    band = (dist > 0) & (dist < reach) & ~(bl > 0) & ~sc.water
    deb = core.rubble_base(sc, rng, sc.f) * 0.85
    alpha = np.clip(1 - dist / np.maximum(reach, 1), 0, 1) ** 0.6 * band
    return res, (alpha.astype(np.float32), deb, band & (alpha > 0.3))


def _tornado_large(sc, rng, sl, s, u, v, orig, cur, cxi, cyi, severity, noise, bl, f):
    """Tornade sur un grand bâtiment à couverture en tôle : TOUT le toit est
    touché, avec une intensité qui varie (plus fort du côté où arrive le vent).
      - couverture arrachée par BANDES de tôles entières (le vent soulève les
        bacs le long des nervures, à partir d'un bord) -> charpente apparente
        (pannes + fermes) au-dessus de l'intérieur ;
      - tôles restantes soulevées / pliées (éclairage différent), décalées ;
      - bacs arrachés éparpillés au sol sous le vent, pliés, avec ombres ;
    Résultat : même vu de loin, le bâtiment est nettement endommagé."""
    h, w = s.shape
    rib = _rib_angle(orig, s)
    x_loc, y_loc = sc.xx[sl] - cxi, sc.yy[sl] - cyi
    sid, a_rib, b_rib = _sheets(sc, x_loc, y_loc, rib, len_m=8.0, wid_m=1.0)
    m = int(sid.max()) + 1
    # direction du vent ; intensité croissante du côté au vent
    th = rng.uniform(0, 2 * np.pi)
    wx, wy = np.cos(th), np.sin(th)
    proj = x_loc * wx + y_loc * wy
    pr = (proj - proj[s].min()) / (np.ptp(proj[s]) + 1e-6)
    inten = np.clip(0.35 + 0.65 * (1 - pr) + (noise(15, 3) - 0.5) * 0.5, 0, 1)
    # probabilité qu'une tôle soit arrachée : par BANDE (colonnes de tôles le
    # long des nervures, corrélées) et selon l'intensité locale
    band = np.floor(b_rib / sc.m(1.0)).astype(np.int64)
    band -= band.min()
    nbnd = int(band.max()) + 1
    band_p = np.clip(core.noise_px((1, nbnd), rng, 4.0, 2).ravel() * 1.3 - 0.15, 0, 1)
    cnt = np.maximum(np.bincount(sid.ravel(), minlength=m), 1)
    s_int = np.bincount(sid.ravel(), inten.ravel(), m) / cnt
    s_band = (np.bincount(sid.ravel(), band.ravel(), m) // cnt).astype(np.int64)
    # arrachement par GRANDES PLAQUES : un champ continu (bruit à ~12 m,
    # renforcé du côté au vent et le long de certaines bandes de nervures)
    # seuillé, puis aligné sur les tôles entières -> zones arrachées d'un
    # seul tenant, et non un damier de tôles isolées
    # toiture PARTIELLEMENT envolée : une zone d'un seul tenant, partant du
    # bord au vent ; le côté sous le vent garde sa couverture
    field = 0.6 * (1 - pr) + 0.3 * noise(12, 3) + 0.1 * band_p[np.clip(band, 0, nbnd - 1)]
    frac_gone = float(np.clip(0.15 + 0.6 * severity, 0.2, 0.5))
    thr = np.quantile(field[s], 1 - frac_gone) if s.any() else 1.0
    zone = field > thr
    z_share = np.bincount(sid.ravel(), zone.ravel().astype(np.float64), m) / cnt
    gone = ((z_share > 0.5) | ((z_share > 0.15) & (rng.random(m) < 0.4)))[sid] & s
    # bords déchiquetés des zones arrachées
    rag = cv2.dilate(gone.astype(np.uint8), core.disk(max(1, sc.m(0.3)))).astype(bool) & s
    gone |= rag & (noise(0.6, 2) > 0.55)

    # intérieur visible : sol / stock poussiéreux sombre + charpente métallique
    inner = (orig * 0.3 + np.array([72, 70, 66], np.float32) * 0.7) * (0.65 + 0.45 * noise(2, 3))[..., None]
    trusses, purlins = _grid(u, v, sc, BAY_M, 0.35, 1.5, 0.12)
    frame = (trusses | purlins) & gone
    steel = np.array([158, 156, 150], np.float32) * (0.8 + 0.3 * noise(0.8, 2))[..., None]
    res = np.where(gone[..., None], inner, cur)
    sh = max(1, int(round(sc.m(0.35))))
    shd = np.roll(np.roll(frame, sh, 0), sh, 1) & gone & ~frame
    res = np.where(shd[..., None], res * 0.55, res)
    res = np.where(frame[..., None], steel, res)

    # tôles restantes : soulevées / pliées -> éclairage et léger décalage par tôle
    keep = s & ~gone
    lift = rng.uniform(0.72, 1.18, m).astype(np.float32)
    # tôles déformées : surtout au bord des zones arrachées (on les voit se
    # soulever et se plier) ; ailleurs, quelques-unes seulement
    near = cv2.dilate(gone.astype(np.uint8), core.disk(sc.m(3))).astype(np.float64)
    s_near = np.bincount(sid.ravel(), near.ravel(), m) / cnt
    lifted = (rng.random(m) < 0.1 + 0.7 * s_near)
    shade = np.where(lifted, lift, 1.0).astype(np.float32)[sid]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dmag = np.where(lifted, np.abs(rng.normal(0, sc.m(0.25), m)), 0)[sid]
    mx = (xx - wx * dmag).astype(np.float32)
    my = (yy - wy * dmag).astype(np.float32)
    shifted = cv2.remap(orig.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    res = np.where(keep[..., None], shifted * shade[..., None], res)
    # ombre des tôles soulevées sur l'intérieur, au bord des trous
    o = sc.m(0.5)
    sh2 = core.shift(keep.astype(np.float32), o * 0.7, o * 0.7) > 0.5
    res = np.where((sh2 & gone & ~frame)[..., None], res * 0.6, res)
    kk = np.ones((3, 3), np.uint8)
    sf = sid.astype(np.float32)
    joint = (cv2.dilate(sf, kk) != cv2.erode(sf, kk)) & keep & lifted[sid]
    res = np.where(joint[..., None], res * 0.7, res)

    # bacs arrachés éparpillés au sol sous le vent (rectangles allongés, pliés)
    canvas = np.zeros((sc.h, sc.w, 3), np.float32)
    pm = np.zeros((sc.h, sc.w), np.uint8)
    shm = np.zeros((sc.h, sc.w), np.uint8)
    y0, x0 = sl[0].start, sl[1].start
    ys, xs = np.nonzero(gone)
    n_deb = int(min(250, len(ys) / max(1.0, sc.m(6)) ** 2))
    if n_deb and len(ys):
        cols = orig[ys, xs]
        for _ in range(n_deb):
            k = rng.integers(len(ys))
            d = sc.m(rng.uniform(3, 30))
            ang = th + rng.normal(0, 0.5)
            cx_, cy_ = x0 + xs[k] + np.cos(ang) * d, y0 + ys[k] + np.sin(ang) * d
            L, W = sc.m(rng.uniform(1.5, 6)), sc.m(rng.uniform(0.6, 1.1))
            rot = np.rad2deg(rib + rng.normal(0, 0.8))
            box = cv2.boxPoints(((cx_, cy_), (L, W), rot)).astype(np.int32)
            col = cols[k] * rng.uniform(0.65, 1.15)
            cv2.fillConvexPoly(shm, box + int(sc.m(0.3)), 255)
            cv2.fillConvexPoly(canvas, box, tuple(float(c) for c in col))
            cv2.fillConvexPoly(pm, box, 255)
    free = ~(bl > 0) & ~sc.water
    piece = (pm > 0) & free
    alpha = np.zeros((sc.h, sc.w), np.float32)
    deb_img = f.copy()
    deb_img = np.where(((shm > 0) & ~piece & free)[..., None], deb_img * 0.6, deb_img)
    deb_img = np.where(piece[..., None], canvas * (0.85 + 0.3 * sc.noise(rng, 0.5, 2))[..., None], deb_img)
    alpha[(shm > 0) & free] = 1.0
    return res, (alpha, deb_img, piece)


def render_large_buildings(sc, kind, out, dmg, rng, severity):
    """Remplace le rendu des grands bâtiments endommagés par un rendu par travées."""
    # inondation : rendu déjà adapté ; séisme : tas de gravats généraux
    if kind in ("flood", "earthquake", "tornado") or not hasattr(sc, "building_labels"):
        return out, dmg
    bl = sc.building_labels
    nb = int(bl.max()) + 1
    if nb <= 1:
        return out, dmg
    area, cx, cy, ang = _building_frames(sc)
    f = sc.f
    out = out.astype(np.float32)
    hit = np.bincount(bl.ravel(), dmg.ravel().astype(np.float64), nb) / np.maximum(
        np.bincount(bl.ravel(), minlength=nb), 1)
    # cohérence image / masque : un bâtiment touché sur moins de
    # DAMAGED_FRACTION de sa surface est "intact" dans le masque -> on lui rend
    # son aspect d'origine (sinon tache de dégâts sur un bâtiment dit intact)
    faint = (hit > 0) & (hit < DAMAGED_FRACTION)
    faint[0] = False
    if faint.any():
        fsel = faint[bl] & (bl > 0)
        out[fsel] = f[fsel]
        dmg[fsel] = False
    for i in range(1, nb):
        if area[i] < LARGE_M2 or hit[i] < DAMAGED_FRACTION:
            continue
        sel = bl == i
        ys, xs = np.nonzero(sel)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sl = (slice(y0, y1), slice(x0, x1))
        s = sel[sl]
        u, v = _frame(sc, sel, cx[i], cy[i], ang[i])
        u, v = u[sl], v[sl]
        bay = np.floor((u - u[s].min()) / sc.m(BAY_M)).astype(np.int32)
        bay_ids = np.unique(bay[s])
        # part de chaque travée atteinte par le simulateur (pour le feu / la tornade)
        dsub = dmg[sl] & s
        prefer = np.array([dsub[bay == b].mean() if (bay == b).any() else 0 for b in sorted(bay_ids)])
        orig = f[sl]
        cur = out[sl].copy()
        cur[s] = orig[s]                                   # repart du toit intact
        tex = sc.tex[sl]
        noise = lambda m_, o=2: sc.noise(rng, m_, o)[sl]

        if kind == "earthquake":
            cur, spill = _quake_large(sc, rng, i, sl, s, u, v, bay, bay_ids, orig, cur,
                                      cx[i], cy[i], severity, noise, bl)
            out[sl] = np.where(s[..., None], cur, out[sl])
            # gravats tombés au pied des murs (hors bâtiment)
            a_sp = spill[0][..., None]
            out = out * (1 - a_sp) + spill[1] * a_sp
            dmg |= spill[2]
            dmg[sl] = dmg[sl] | s
            continue

        elif kind in ("fire", "tornado"):
            if kind == "tornado":
                cur, spill = _tornado_large(sc, rng, sl, s, u, v, orig, cur, cx[i], cy[i],
                                            severity, noise, bl, f)
                out[sl] = np.where(s[..., None], cur, out[sl])
                a_sp = spill[0][..., None]
                out = out * (1 - a_sp) + spill[1] * a_sp
                dmg |= spill[2]
                dmg[sl] = dmg[sl] | s
                continue
            frac = float(np.clip(0.25 + 0.9 * severity, 0.3, 0.95))
            chosen = _pick_bays(bay, bay_ids, frac, rng, prefer)
            R = s & np.isin(bay, list(chosen))
            trusses, purlins = _grid(u, v, sc, BAY_M, 0.35, 1.5, 0.12)
            if kind == "fire":
                # tôles drapées sur l'intérieur effondré : couverture d'origine
                # (nervures conservées) noircie / rouillée, affaissée et gondolée,
                # trouée par endroits sur un fond carbonisé où l'on voit les fermes
                burnt = core.luminance(orig)[..., None] * 0.38 + np.array([24, 20, 17], np.float32)
                rust = noise(3, 3)[..., None]
                burnt = burnt * (1 - 0.35 * rust) + np.array([96, 58, 38], np.float32) * 0.35 * rust
                sag = _sagged_roof(sc, rng, burnt, u, v, R, noise, depth=0.35)
                char = np.array([30, 28, 26], np.float32) * (0.7 + 0.6 * noise(1.0, 3))[..., None]
                char = core.blend(char, np.array([118, 114, 108], np.float32),
                                  np.clip((noise(2.0, 3) - 0.7) * 2, 0, 0.35))      # cendres
                # la plupart des tôles sont tombées dans le brasier ; celles qui
                # restent pendent, noircies et rouillées, par groupes
                rib = _rib_angle(orig, s)
                sid, _, _ = _sheets(sc, sc.xx[sl] - cx[i], sc.yy[sl] - cy[i], rib)
                p_miss = (0.45 + 0.4 * noise(8, 2)) * R
                cur, holes = _sheet_damage(sc, rng, cur, orig, R, sid, p_miss, char, sag, 0.25, noise)
                steel = np.array([92, 62, 46], np.float32) * (0.75 + 0.4 * noise(0.8, 2))[..., None]
                frame = (trusses & holes) | (purlins & holes & (noise(1.5, 2) > 0.5))
                sh = max(1, int(round(sc.m(0.3))))
                shadow = np.roll(np.roll(frame, sh, 0), sh, 1) & holes & ~frame
                cur = np.where(shadow[..., None], cur * 0.6, cur)
                cur = np.where(frame[..., None], steel, cur)
                d = cv2.distanceTransform((~R).astype(np.uint8), cv2.DIST_L2, 3) / sc.ppm
                heat = s & ~R & (d < 4)
                t = np.clip(1 - d / 4, 0, 1)[..., None] * (0.6 + 0.4 * noise(1.5, 2))[..., None]
                cur = np.where(heat[..., None], orig * (1 - t) + orig * np.array([0.55, 0.42, 0.36], np.float32) * t, cur)
                out[sl] = np.where(s[..., None], cur, out[sl])
                dmg[sl] = dmg[sl] | s
                continue
            else:
                inner = (orig * 0.35 + np.array([70, 68, 64], np.float32) * 0.65) * \
                        (0.75 + 0.4 * noise(1.5, 3))[..., None]
                steel = np.array([150, 150, 146], np.float32) * (0.8 + 0.3 * noise(0.8, 2))[..., None]
                keep_sheet = 0.35
            cur = np.where(R[..., None], inner, cur)
            # charpente apparente (fermes + pannes), avec ombre portée
            frame = (trusses | purlins) & R
            sh = max(1, int(round(sc.m(0.3))))
            shadow = np.roll(np.roll(frame, sh, 0), sh, 1) & R & ~frame
            cur = np.where(shadow[..., None], cur * 0.55, cur)
            cur = np.where(frame[..., None], steel, cur)
            # quelques tôles restantes, tordues (et brûlées pour le feu)
            left = R & (noise(4, 2) > 1 - keep_sheet)
            src = orig if kind == "tornado" else (core.luminance(orig)[..., None] * 0.45 +
                                                  np.array([30, 22, 18], np.float32))
            p, val, jn = _panels(sc, rng, left, src, u, v, BAY_M / 2, 2.5, 0.3, 0.45, 0.25)
            sh = max(1, int(round(sc.m(0.25))))
            shadow = np.roll(np.roll(val, sh, 0), sh, 1) & R & ~val
            cur = np.where(shadow[..., None], cur * 0.55, cur)
            cur = np.where(val[..., None], p, cur)
            cur = np.where(jn[..., None], cur * 0.55, cur)
            if kind == "fire":
                # tôles rouillées / noircies par la chaleur à côté des travées brûlées
                d = cv2.distanceTransform((~R).astype(np.uint8), cv2.DIST_L2, 3) / sc.ppm
                heat = s & ~R & (d < 4)
                t = np.clip(1 - d / 4, 0, 1)[..., None] * (0.6 + 0.4 * noise(1.5, 2))[..., None]
                scorched = orig * np.array([0.55, 0.42, 0.36], np.float32)
                cur = np.where(heat[..., None], orig * (1 - t) + scorched * t, cur)
        else:
            continue
        out[sl] = np.where(s[..., None], cur, out[sl])
        # masque : le bâtiment reste "endommagé" en entier (convention RescueNet)
        dmg[sl] = dmg[sl] | s
    return np.clip(out, 0, 255).astype(np.uint8), dmg


# ---------------------------------------------------------------------------
# Masque au style RescueNet
# ---------------------------------------------------------------------------
def rescuenet_mask(scene, damage):
    """Fond / route / bâtiment intact / bâtiment endommagé.
    Un bâtiment est endommagé si au moins DAMAGED_FRACTION de sa surface est
    dans la zone de dégâts du simulateur (RescueNet annote par bâtiment)."""
    out = np.zeros(scene.lab.shape + (3,), np.uint8)
    out[scene.road_px] = MASK_COLORS["route"]
    bl = scene.building_labels
    nb = int(bl.max()) + 1
    n_dmg = 0
    if nb > 1:
        tot = np.bincount(bl.ravel(), minlength=nb).astype(np.float64)
        hit = np.bincount(bl.ravel(), weights=damage.ravel().astype(np.float64), minlength=nb)
        damaged = hit / np.maximum(tot, 1) >= DAMAGED_FRACTION
        damaged[0] = False
        fp = _full_footprints(bl, scene.ppm)      # maison entière, une seule couleur
        out[(fp > 0) & ~damaged[fp]] = MASK_COLORS["batiment_intact"]
        out[(fp > 0) & damaged[fp]] = MASK_COLORS["batiment_endommage"]
        n_dmg = int(damaged[1:].sum())
    return out, n_dmg, max(0, nb - 1)


def legend(tile):
    s = tile.shape[1] / 600
    x, y = int(8 * s), tile.shape[0] - int(10 * s)
    for name, col in [("intact", MASK_COLORS["batiment_intact"]),
                      ("endommage", MASK_COLORS["batiment_endommage"]),
                      ("route", MASK_COLORS["route"])]:
        cv2.rectangle(tile, (x, y - int(16 * s)), (x + int(16 * s), y), col, -1)
        cv2.rectangle(tile, (x, y - int(16 * s)), (x + int(16 * s), y), (255, 255, 255), 1)
        cv2.putText(tile, name, (x + int(22 * s), y - int(2 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, (255, 255, 255), max(1, int(1 * s)), cv2.LINE_AA)
        x += int((40 + 11 * len(name)) * s)
    return tile


def planche(sain, out, mask, kind, panel=600):
    h = int(panel * sain.shape[0] / sain.shape[1])
    rs = lambda a, i=cv2.INTER_AREA: cv2.resize(a, (panel, h), interpolation=i)
    return np.hstack([core.label(rs(sain), "saine"), core.label(rs(out), kind),
                      legend(core.label(rs(mask, cv2.INTER_NEAREST), "masque (style RescueNet)"))])


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
def _reduce_result(img, hi, out, mask):
    """Ramène une simulation suréchantillonnée à la taille de la photo : les
    bords des débris sont lissés, et les pixels non touchés restent ceux d'origine."""
    h, w = img.shape[:2]
    changed = (out != hi).any(2).astype(np.float32)
    a = cv2.resize(changed, (w, h), interpolation=cv2.INTER_AREA)[..., None]
    small = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    merged = img.astype(np.float32) * (1 - a) + small * a
    return (np.clip(np.rint(merged), 0, 255).astype(np.uint8),
            cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST))


_INT_SCENES = {}        # scène construite une fois par processus de calcul


def _intensity_scene(prof, image_id, width, supersample):
    key = (prof.name, image_id, width, supersample)
    if key not in _INT_SCENES:
        img, lab = load_drone(prof, image_id, width)
        ppm, _ = estimate_ppm(lab, prof.car, prof.default_ppm_full * img.shape[1]
                              / float(getattr(prof, "full_width", 6000)))
        hi, lab_hi, ppm_hi = img, lab, ppm
        if supersample > 1:
            size = (img.shape[1] * supersample, img.shape[0] * supersample)
            hi = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
            lab_hi = cv2.resize(lab, size, interpolation=cv2.INTER_NEAREST)
            ppm_hi = ppm * supersample
        inst = prof.instance_map(image_id, lab_hi.shape) if hasattr(prof, "instance_map") else None
        _INT_SCENES[key] = (img, hi, DroneScene(hi, lab_hi, ppm_hi, prof, inst))
    return _INT_SCENES[key]


def _intensity_level(job):
    """Calcule un niveau d'intensité (appelé dans un processus séparé ou non)."""
    global USE_REAL_TEX, USE_TORNADO_TEX, NATURAL_PIECES
    pname, root, image_id, width, supersample, seed, x, flags, out_dir, panel = job
    USE_REAL_TEX, USE_TORNADO_TEX, NATURAL_PIECES = flags
    prof = PROFILES[pname]()
    if root:
        prof.root = root
    img, hi, scene = _intensity_scene(prof, image_id, width, supersample)
    out, dmg, level = simulate_tornado_intensity(scene, seed, x)
    # masque style RescueNet : un bâtiment endommagé est rouge, quel que soit son niveau
    mask, n_dmg, n_bat = rescuenet_mask(scene, dmg)
    if supersample > 1:
        out, mask = _reduce_result(img, hi, out, mask)
    tag = "%s_tornado_i%03d" % (image_id, int(round(100 * x)))
    core.write_rgb(os.path.join(out_dir, tag + ".jpg"), out)
    cv2.imwrite(os.path.join(out_dir, tag + "_masque.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    h = int(panel * img.shape[0] / img.shape[1])
    counts = [int((level == k).sum()) for k in (1, 2, 3)]
    return (x, n_bat, n_dmg, counts,
            cv2.resize(out, (panel, h), interpolation=cv2.INTER_AREA),
            cv2.resize(mask, (panel, h), interpolation=cv2.INTER_NEAREST))


def _level_legend(tile):
    s = tile.shape[1] / 480
    x, y = int(6 * s), tile.shape[0] - int(8 * s)
    for name, k in (("intact", 0), ("peu", 1), ("partiel", 2), ("detruit", 3)):
        col = LEVEL_COLORS[k]
        cv2.rectangle(tile, (x, y - int(13 * s)), (x + int(13 * s), y), col, -1)
        cv2.rectangle(tile, (x, y - int(13 * s)), (x + int(13 * s), y), (255, 255, 255), 1)
        cv2.putText(tile, name, (x + int(17 * s), y - int(2 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42 * s, (255, 255, 255), 1, cv2.LINE_AA)
        x += int((30 + 8 * len(name)) * s)
    return tile


def process_intensity(prof, image_id, width, levels, seed, out_dir, supersample=1, workers=None):
    """Tornade sur une même image à plusieurs intensités : une image et un masque
    (style RescueNet : intact, endommagé en rouge, route) par intensité, plus une
    planche d'évolution. Les niveaux sont calculés en parallèle."""
    img, _ = load_drone(prof, image_id, width)
    core.write_rgb(os.path.join(out_dir, "%s_sain.jpg" % image_id), img)
    panel = 480
    root = prof.root if prof.root != type(prof).root else None
    jobs = [(prof.name, root, image_id, width, supersample, seed, float(x),
             (USE_REAL_TEX, USE_TORNADO_TEX, NATURAL_PIECES), out_dir, panel) for x in levels]
    if workers is None:
        workers = min(len(jobs), 4 if supersample > 1 else 6, max(1, (os.cpu_count() or 2) - 1))
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_intensity_level, jobs))
    else:
        results = [_intensity_level(j) for j in jobs]
    top, bottom, rows = [], [], []
    for x, n_bat, n_dmg, (n_l, n_p, n_f), small, lsmall in results:
        top.append(core.label(small, "intensite %.2f" % x))
        bottom.append(legend(core.label(lsmall, "%d/%d endommages" % (n_dmg, n_bat))))
        print("%s intensite=%.2f  batiments endommages %d/%d  (peu touches %d, partiels %d, detruits %d)"
              % (image_id, x, n_dmg, n_bat, n_l, n_p, n_f), flush=True)
        rows.append([image_id, "tornado", round(x, 3), n_bat, n_dmg, n_l, n_p, n_f, seed])
    core.write_rgb(os.path.join(out_dir, "%s_evolution.jpg" % image_id),
                   np.vstack([np.hstack(top), np.hstack(bottom)]))
    with open(os.path.join(out_dir, "%s_evolution.csv" % image_id), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "desastre", "intensite", "batiments", "batiments_endommages",
                     "peu_touches", "partiels", "detruits", "graine"])
        wr.writerows(rows)
    return rows


def process(prof, image_id, width, kinds, rng, severity, out_dir, write_planche=True, supersample=1):
    img, lab = load_drone(prof, image_id, width)
    ppm, src = estimate_ppm(lab, prof.car, prof.default_ppm_full * img.shape[1]
                            / float(getattr(prof, "full_width", 6000)))
    hi, lab_hi, ppm_hi = img, lab, ppm
    if supersample > 1:                     # simulation plus grande, réduite ensuite
        size = (img.shape[1] * supersample, img.shape[0] * supersample)
        hi = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        lab_hi = cv2.resize(lab, size, interpolation=cv2.INTER_NEAREST)
        ppm_hi = ppm * supersample
    inst = prof.instance_map(image_id, lab_hi.shape) if hasattr(prof, "instance_map") else None
    scene = DroneScene(hi, lab_hi, ppm_hi, prof, inst)
    core.write_rgb(os.path.join(out_dir, "%s_sain.jpg" % image_id), img)
    rows = []
    for kind in kinds:
        out, dmg, sev = core.apply_disaster(scene, kind, rng, severity)
        out, dmg = render_large_buildings(scene, kind, out, dmg, rng, sev)
        mask, n_dmg, n_bat = rescuenet_mask(scene, dmg)
        if supersample > 1:
            out, mask = _reduce_result(img, hi, out, mask)
        core.write_rgb(os.path.join(out_dir, "%s_%s.jpg" % (image_id, kind)), out)
        cv2.imwrite(os.path.join(out_dir, "%s_%s_masque.png" % (image_id, kind)),
                    cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        if write_planche == "dossiers":       # mode dataset : planches/<desastre>/<id>.jpg
            core.write_rgb(os.path.join(out_dir, "planches", kind, "%s.jpg" % image_id),
                           planche(img, out, mask, kind))
        elif write_planche:
            core.write_rgb(os.path.join(out_dir, "%s_%s_planche.jpg" % (image_id, kind)),
                           planche(img, out, mask, kind))
        print("%s %-10s severite=%.2f  batiments endommages %d/%d  (echelle %.1f px/m, %s)"
              % (image_id, kind, sev, n_dmg, n_bat, ppm, src))
        rows.append([image_id, kind, round(sev, 3), n_bat, n_dmg, round(ppm, 2), src])
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Images de drone saines -> avec desastre + masque style RescueNet")
    p.add_argument("--source", default="vdd", choices=sorted(PROFILES))
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="identifiant d'image (ex. DJI_0739 pour VDD, 259 pour SDD)")
    mode.add_argument("--dataset", action="store_true", help="traiter la selection d'images")
    mode.add_argument("--list", action="store_true", help="VDD : afficher les images retenues")
    p.add_argument("--root", default=None, help="dossier du dataset (defaut selon --source)")
    p.add_argument("--type", default="all", choices=("all",) + core.DISASTERS)
    p.add_argument("--severity", type=float, default=None)
    p.add_argument("--width", type=int, default=None, help="largeur de travail (defaut : 2000 VDD, 1500 SDD)")
    p.add_argument("--n", type=int, default=None, help="mode dataset : nombre d'images")
    p.add_argument("--min-alt", type=float, default=None,
                   help="VDD/UDD : altitude estimee minimale en metres (defaut 85 VDD, 0 UDD)")
    p.add_argument("--min-roof", type=float, default=0.05,
                   help="SDD : part minimale de toit pour garder une image")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--sans-textures", action="store_true",
                   help="gravats 100%% proceduraux (sans les textures reelles)")
    p.add_argument("--textures-tornade", action="store_true",
                   help="tornade : textures de vrais degats de tornade (textures_tornade/)")
    p.add_argument("--morceaux-naturels", action="store_true",
                   help="morceaux de toit a bords irreguliers, sans contour noir")
    p.add_argument("--intensite", type=float, default=None,
                   help="tornade a intensite reglable (0 a 1), meme trajectoire quelle que soit la valeur ; avec --image")
    p.add_argument("--evolution", type=int, default=None,
                   help="tornade : N intensites de 0 a 1 sur la meme image, avec planche d'evolution ; avec --image")
    p.add_argument("--processus", type=int, default=None,
                   help="--evolution : nombre de niveaux calcules en parallele (defaut : automatique)")
    p.add_argument("--supersample", type=int, default=1,
                   help="simule a N fois la taille puis reduit, pour lisser les bords (defaut 1 : desactive)")
    p.add_argument("--planches", action="store_true",
                   help="mode dataset : ecrire aussi les planches dans planches/<desastre>/")
    args = p.parse_args(argv)

    global USE_REAL_TEX, USE_TORNADO_TEX, NATURAL_PIECES
    USE_REAL_TEX = not args.sans_textures
    USE_TORNADO_TEX = args.textures_tornade
    NATURAL_PIECES = args.morceaux_naturels
    prof = PROFILES[args.source]()
    if args.min_alt is None:
        args.min_alt = getattr(prof, "default_min_alt", 85.0)
    if args.root:
        prof.root = args.root
    width = args.width or prof.default_width
    out_dir = args.out or ("sortie_" + prof.name)
    rng = np.random.default_rng(args.seed)
    kinds = core.DISASTERS if args.type == "all" else (args.type,)

    if args.list:
        if prof.name not in ("vdd", "udd"):
            sys.exit("--list ne concerne que VDD et UDD")
        sel = prof.selection(args.min_alt)
        for image_id, alt, ppm in sel:
            if alt is None:
                print("%-10s pas de voiture : echelle mediane %.1f px/m (~%.0f m)"
                      % (image_id, prof.default_ppm_full, prof.alt_k / prof.default_ppm_full))
            else:
                print("%-10s altitude estimee ~%3.0f m  (%.1f px/m)" % (image_id, alt, ppm))
        n_alt = sum(t[1] is not None for t in sel)
        print("%d images retenues : %d avec altitude estimee >= %.0f m + %d vues verticales sans voiture"
              % (len(sel), n_alt, args.min_alt, len(sel) - n_alt))
        return

    if args.intensite is not None or args.evolution is not None:
        if not args.image:
            sys.exit("--intensite et --evolution s'utilisent avec --image")
        if args.evolution is not None and args.evolution < 2:
            sys.exit("--evolution demande au moins 2 niveaux")
        levels = ([args.intensite] if args.evolution is None
                  else [float(v) for v in np.linspace(0, 1, args.evolution)])
        seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2 ** 31))
        os.makedirs(out_dir, exist_ok=True)
        process_intensity(prof, args.image, width, levels, seed, out_dir, supersample=args.supersample,
                          workers=args.processus)
        print("Graine %d (a reutiliser avec --seed pour la meme trajectoire)" % seed)
        print("Resultats ecrits dans", os.path.abspath(out_dir))
        return

    os.makedirs(out_dir, exist_ok=True)
    if args.image:
        process(prof, args.image, width, kinds, rng, args.severity, out_dir, supersample=args.supersample)
        print("Resultats ecrits dans", os.path.abspath(out_dir))
        return

    if prof.name in ("vdd", "udd"):
        sel = prof.selection(args.min_alt)
        ids = [t[0] for t in sel]
        print(("%d images " + prof.name.upper() + " retenues (%d avec altitude estimee >= %.0f m, %d sans voiture)")
              % (len(ids), sum(t[1] is not None for t in sel), args.min_alt,
                 sum(t[1] is None for t in sel)))
    elif prof.name == "instancebuilding":
        ids = prof.ids()
        print("%d images verticales avec au moins %d toits" % (len(ids), prof.min_polygons))
    elif prof.name == "okutama":
        ids = prof.ids()
        print("%d images Okutama + Swiss" % len(ids))
    else:
        ids = [i for i in prof.ids()
               if (cv2.imread(prof.paths(i)[1], cv2.IMREAD_UNCHANGED)[::16, ::16]
                   == prof.C["roof"]).mean() >= args.min_roof]
        print("%d images SDD avec au moins %.0f%% de toit" % (len(ids), 100 * args.min_roof))
    if args.n:
        ids = [ids[i] for i in rng.permutation(len(ids))[: args.n]]
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "desastre", "severite", "batiments", "batiments_endommages",
                     "px_par_m", "source_echelle"])
        for i, image_id in enumerate(ids, 1):
            for row in process(prof, image_id, width, kinds, rng, args.severity, out_dir,
                               write_planche="dossiers" if args.planches else False,
                               supersample=args.supersample):
                wr.writerow(row)
            print("[%d/%d]" % (i, len(ids)))
    print("Dataset ecrit dans", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
