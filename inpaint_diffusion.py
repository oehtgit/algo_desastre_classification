# -*- coding: utf-8 -*-
"""
inpaint_diffusion.py
====================

Méthode hybride "procédural + diffusion" :
  1. algo_desastre_drone.py décide QUELS bâtiments sont touchés, dessine un
     premier rendu procédural et produit le masque (exact) ;
  2. Stable Diffusion (inpainting, modèle pré-entraîné généraliste, AUCUNE
     donnée RescueNet) repeint uniquement l'emprise de chaque bâtiment touché,
     en partant du rendu procédural (img2img partiel) : textures réalistes,
     géométrie et labels inchangés.

Hors du masque, l'image n'est pas modifiée.

Usage :
    python inpaint_diffusion.py --image DJI_0679 --type earthquake --seed 0
    python inpaint_diffusion.py --dataset                 # séisme + tornade -> dataset_final/
"""

import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("drone", os.path.join(HERE, "algo_desastre_drone.py"))
drone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drone)
core = drone.core

MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
BASE = ("aerial photograph, top-down nadir drone view, high resolution, realistic, "
        "natural daylight, ")
PROMPTS = {
    "earthquake": "collapsed building after an earthquake, pile of concrete rubble, broken slabs, "
                  "crushed debris, dust",
    "tornado": "house heavily damaged by a tornado, roof torn off, exposed wooden rafters, "
               "missing shingles, scattered debris",
}
NEGATIVE = ("intact roof, undamaged house, clean roof, new building, drawing, painting, cartoon, map, text, watermark, blurry, oblique view, perspective, "
            "people, car, sky, low quality")
SIZE = 512


LOG = os.path.join(HERE, "progression.log")


def log(msg):
    """Affiche et ajoute au fichier progression.log (à ouvrir pour suivre)."""
    import datetime
    line = "%s  %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_pipe():
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def inpaint_buildings(pipe, scene, kind, out, dmg, seed, strength, steps):
    """Repeint chaque bâtiment endommagé (selon le masque) dans une fenêtre
    carrée autour de lui, ramenée à 512 px."""
    import torch
    if kind not in PROMPTS:
        return out, 0
    bl = scene.building_labels
    nb = int(bl.max()) + 1
    tot = np.bincount(bl.ravel(), minlength=nb).astype(np.float64)
    hit = np.bincount(bl.ravel(), dmg.ravel().astype(np.float64), nb) / np.maximum(tot, 1)
    ids = [i for i in range(1, nb) if hit[i] >= drone.DAMAGED_FRACTION]
    res = out.astype(np.float32).copy()
    h, w = bl.shape
    n_done = 0
    import time
    t0 = time.time()
    for k, i in enumerate(ids):
        if k:
            el = time.time() - t0
            log("  %s : batiment %d/%d  (%.1f s/batiment, reste ~%.0f s)"
                % (kind, k + 1, len(ids), el / k, el / k * (len(ids) - k)))
        ys, xs = np.nonzero(bl == i)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        side = int(max(ys.ptp(), xs.ptp()) * 1.5) + 2 * int(scene.m(4))
        side = int(np.clip(side, 128, min(h, w)))
        y0 = int(np.clip(cy - side / 2, 0, h - side))
        x0 = int(np.clip(cx - side / 2, 0, w - side))
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        foot = (bl[sl] == i).astype(np.uint8)
        # zone repeinte : l'emprise + ~0,6 m (bords de gravats)
        grow = cv2.dilate(foot, core.disk(max(1, scene.m(0.6))))
        crop = np.clip(res[sl], 0, 255).astype(np.uint8)
        img512 = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        m512 = cv2.resize(grow * 255, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        from PIL import Image
        g = torch.Generator("cuda").manual_seed(seed * 1000 + k)
        gen = pipe(prompt=BASE + PROMPTS[kind], negative_prompt=NEGATIVE,
                   image=Image.fromarray(img512), mask_image=Image.fromarray(m512),
                   height=SIZE, width=SIZE, strength=strength, num_inference_steps=steps,
                   guidance_scale=7.0, generator=g).images[0]
        gen = cv2.resize(np.asarray(gen), (side, side), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # retour dans l'image : seulement dans la zone repeinte, bord fondu
        a = cv2.GaussianBlur(grow.astype(np.float32), (0, 0), max(1.0, scene.m(0.15)))
        a = np.maximum(a, foot.astype(np.float32))[..., None]
        res[sl] = res[sl] * (1 - a) + gen * a
        n_done += 1
    return np.clip(res, 0, 255).astype(np.uint8), n_done


def process_image(pipe, prof, image_id, kinds, seed, a, out_dir, layout):
    """Une image : sain + (procédural -> diffusion) par désastre + masque + planche.
    layout "plat" : tout dans out_dir ; "dataset" : sous-dossiers par type."""
    img, lab = drone.load_drone(prof, image_id, prof.default_width)
    ppm, _ = drone.estimate_ppm(lab, prof.car, prof.default_ppm_full * img.shape[1] / 4000.0)
    scene = drone.DroneScene(img, lab, ppm, prof)
    rng = np.random.default_rng(seed)
    if layout == "dataset":
        P = lambda sub, name: os.path.join(out_dir, sub, name)
        p_sain = P("sain", "%s.jpg" % image_id)
    else:
        P = lambda sub, name: os.path.join(out_dir, name)
        p_sain = P("", "%s_sain.jpg" % image_id)
    core.write_rgb(p_sain, img)
    rows = []
    for kind in kinds:
        out, dmg, sev = core.apply_disaster(scene, kind, rng, a.severity)
        out, dmg = drone.render_large_buildings(scene, kind, out, dmg, rng, sev)
        mask, n_dmg, n_bat = drone.rescuenet_mask(scene, dmg)
        diff, n = inpaint_buildings(pipe, scene, kind, out, dmg, seed, a.strength, a.steps)
        if layout == "dataset":
            p_img, p_mask = P(kind, "%s.jpg" % image_id), P("masques/" + kind, "%s.png" % image_id)
            p_proc, p_pl = None, P("planches/" + kind, "%s.jpg" % image_id)
        else:
            p_img, p_mask = P("", "%s_%s.jpg" % (image_id, kind)), P("", "%s_%s_masque.png" % (image_id, kind))
            p_proc, p_pl = P("", "%s_%s_procedural.jpg" % (image_id, kind)), P("", "%s_%s_planche.jpg" % (image_id, kind))
        core.write_rgb(p_img, diff)
        os.makedirs(os.path.dirname(p_mask), exist_ok=True)
        cv2.imwrite(p_mask, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        if p_proc:
            core.write_rgb(p_proc, out)
        panel = 600
        hh = int(panel * img.shape[0] / img.shape[1])
        rs = lambda x, it=cv2.INTER_AREA: cv2.resize(x, (panel, hh), interpolation=it)
        sheet = np.hstack([core.label(rs(img), "saine"), core.label(rs(out), "procedural"),
                           core.label(rs(diff), "procedural + diffusion"),
                           drone.legend(core.label(rs(mask, cv2.INTER_NEAREST), "masque"))])
        core.write_rgb(p_pl, sheet)
        log("%s %-10s %d batiments repeints (%d/%d endommages)" % (image_id, kind, n, n_dmg, n_bat))
        rows.append([image_id, kind, round(sev, 3), n_bat, n_dmg, round(ppm, 2)])
    return rows


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="une image VDD (ex. DJI_0679)")
    mode.add_argument("--dataset", action="store_true",
                      help="toutes les images VDD retenues (vues verticales, altitude haute)")
    p.add_argument("--type", default="all", choices=("all",) + tuple(PROMPTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--severity", type=float, default=None)
    p.add_argument("--strength", type=float, default=0.5,
                   help="part de l'image repeinte (0 = garde le procedural, 1 = tout regenere)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--n", type=int, default=None, help="mode dataset : limiter a N images")
    p.add_argument("--min-alt", type=float, default=85.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    prof = drone.VDDProfile()
    kinds = tuple(PROMPTS) if a.type == "all" else (a.type,)
    log("Chargement du modele sur le GPU...")
    pipe = load_pipe()
    log("Modele charge.")

    if a.image:
        out_dir = a.out or "sortie_diffusion"
        os.makedirs(out_dir, exist_ok=True)
        process_image(pipe, prof, a.image, kinds, a.seed, a, out_dir, "plat")
        log("Termine. Resultats dans %s" % os.path.abspath(out_dir))
        return

    import csv
    out_dir = a.out or "dataset_final"
    os.makedirs(out_dir, exist_ok=True)
    ids = [t[0] for t in prof.selection(a.min_alt)]
    if a.n:
        ids = ids[: a.n]
    csv_path = os.path.join(out_dir, "labels.csv")
    done = set()
    if os.path.exists(csv_path):                      # reprise après interruption
        with open(csv_path) as fh:
            done = {r["image"] for r in csv.DictReader(fh)}
    new = not os.path.exists(csv_path)
    log("%d images a traiter (%d deja faites)" % (len(ids), len(done & set(ids))))
    with open(csv_path, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["image", "desastre", "severite", "batiments", "batiments_endommages", "px_par_m"])
        for k, image_id in enumerate(ids, 1):
            if image_id in done:
                continue
            log("[%d/%d] %s" % (k, len(ids), image_id))
            # graine propre à chaque image : résultat reproductible et indépendant de l'ordre
            for row in process_image(pipe, prof, image_id, kinds, a.seed * 100003 + k, a,
                                     out_dir, "dataset"):
                wr.writerow(row)
            fh.flush()
    log("Termine. Dataset dans %s" % os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
