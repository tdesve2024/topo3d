#!/usr/bin/env python3
"""
carte3d.py — Pipeline principal : preview 2D + génération SVG
Port-Blanc — Penvénan (Côtes-d'Armor)

Usage:
  python carte3d.py --preview
  python carte3d.py --generate
  python carte3d.py --preview --png           # export PNG
  python carte3d.py --preview --show-pieces   # colorie chaque pièce
"""
import argparse, math, types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.interpolate import splprep, splev
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape, Polygon, MultiPolygon, box as shapely_box
import svgwrite
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.widgets import Button


# ── Constantes ────────────────────────────────────────────────────────────────
MNT_DEFAUT       = "data/mnt_fusion.tif"
SORTIE_DEFAUT    = "sortie"

COULEURS_MARINE = ['#0d2b52', '#1a3a6c', '#2E6DA4', '#5a9fc0', '#89c4e1', '#b0d9ed',
                   '#c8eaf5', '#ddf2fb']
COULEURS_TERRE  = ['#C8A96E', '#7ab648', '#5D8A3C', '#4a7231', '#3B5E2B', '#2d4a21',
                   '#1e3016', '#8B7355', '#6b5a45', '#4a3d2e', '#2d2419', '#1a1510']


# ── Modèle de couche ──────────────────────────────────────────────────────────
@dataclass
class Couche:
    nom: str
    z_low: float
    z_high: float
    materiau: str          # "mdf" | "plexi"
    couleur: str           # hex


def definir_couches(z_min: float, z_max: float, equi: float) -> List[Couche]:
    couches = []

    # Marine : de z_bottom (arrondi par-dessous) à 0
    z_bottom = math.floor(z_min / equi) * equi
    z_vals_marine = np.arange(z_bottom, 0, equi)
    for i, z in enumerate(z_vals_marine):
        couches.append(Couche(
            nom=f"marine_{int(z):+d}_{int(z + equi):+d}",
            z_low=float(z), z_high=float(z + equi),
            materiau="mdf",
            couleur=COULEURS_MARINE[min(i, len(COULEURS_MARINE) - 1)]
        ))

    # Plexiglass
    couches.append(Couche(nom="plexi_0", z_low=0, z_high=0,
                          materiau="plexi", couleur="#d0eeff"))

    # Terre : de 0 à z_top (arrondi par-dessus)
    z_top = math.ceil(z_max / equi) * equi
    z_vals_terre = np.arange(0, z_top, equi)
    for i, z in enumerate(z_vals_terre):
        couches.append(Couche(
            nom=f"terre_{int(z):+d}_{int(z + equi):+d}",
            z_low=float(z), z_high=float(z + equi),
            materiau="mdf",
            couleur=COULEURS_TERRE[min(i, len(COULEURS_TERRE) - 1)]
        ))

    return couches


# ── Lissage par B-spline ─────────────────────────────────────────────────────
def lisser_polygone_spline(poly: Polygon, lissage_m: float) -> Polygon:
    """
    Lisse un polygone par interpolation B-spline périodique.
    Produit des courbes organiques, adaptées au fraisage CNC.
    lissage_m contrôle la tolérance de lissage (mêmes unités que les coordonnées).
    """
    def _lisser_anneau(coords):
        pts = np.array(coords[:-1])   # retirer le point de fermeture
        n = len(pts)
        if n < 6:
            return coords
        # Périmètre pour calibrer la résolution de sortie
        diffs = np.diff(pts, axis=0, prepend=pts[-1:])
        perimetre = np.sqrt((diffs ** 2).sum(axis=1)).sum()
        # Facteur de lissage scipy : n × tolérance²
        s = n * lissage_m ** 2
        # ~1 point par lissage_m/2, borné entre 30 et 2000
        n_out = max(30, min(2000, int(perimetre / (lissage_m * 0.5))))
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, per=True, k=3)
            u_new = np.linspace(0, 1, n_out)
            xs, ys = splev(u_new, tck)
            result = list(zip(xs.tolist(), ys.tolist()))
            result.append(result[0])   # refermer l'anneau
            return result
        except Exception:
            return coords

    try:
        ext = _lisser_anneau(poly.exterior.coords)
        ints = [_lisser_anneau(r.coords) for r in poly.interiors]
        p = Polygon(ext, ints)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            return poly
        # buffer(0) peut créer un MultiPolygon — garder le plus grand
        if isinstance(p, MultiPolygon):
            p = max(p.geoms, key=lambda g: g.area)
        return p
    except Exception:
        return poly


# ── Extraction de polygones ───────────────────────────────────────────────────
def extraire_polygones(data: np.ma.MaskedArray, z_low: float,
                        raster_transform, seuil_m2: float,
                        simplification_m: float = 5.0,
                        lissage_m: float = 2.0,
                        methode_lissage: str = "spline") -> List[Polygon]:
    """
    Polygones de la zone où altitude >= z_low, filtrés par surface min.
    - simplification_m  : tolérance Douglas-Peucker en mètres réels
    - lissage_m         : rayon/tolérance de lissage en mètres réels (0 = désactivé)
    - methode_lissage   : "spline" (B-spline, courbes organiques) ou "buffer" (double-buffer)
    """
    binary = np.zeros(data.shape, dtype=np.uint8)
    if isinstance(data.mask, np.ndarray):
        valid = ~data.mask
    else:
        valid = np.ones(data.shape, dtype=bool)
    binary[(data.data >= z_low) & valid] = 1

    polys = []
    for geom_dict, val in rasterio_shapes(binary, transform=raster_transform, connectivity=8):
        if val != 1:
            continue
        geom = shape(geom_dict)
        if not geom.is_valid:
            geom = geom.buffer(0)
        # Aplatir MultiPolygon en Polygons individuels
        if isinstance(geom, MultiPolygon):
            candidates = list(geom.geoms)
        else:
            candidates = [geom]
        for poly in candidates:
            if poly.area < seuil_m2:
                continue
            # 1. Simplification Douglas-Peucker
            if simplification_m > 0:
                poly = poly.simplify(simplification_m, preserve_topology=True)
            # 2. Lissage
            if lissage_m > 0:
                if methode_lissage == "spline":
                    poly = lisser_polygone_spline(poly, lissage_m)
                else:
                    poly = poly.buffer(lissage_m).buffer(-lissage_m)
            if poly.is_empty:
                continue
            # Re-filtrer après simplification (peut réduire la surface)
            if isinstance(poly, MultiPolygon):
                candidates2 = list(poly.geoms)
            else:
                candidates2 = [poly]
            for p in candidates2:
                if p.area < seuil_m2:
                    continue
                # 3. Suppression des micro-trous (même seuil que pour les pièces)
                if p.interiors:
                    kept = [h for h in p.interiors
                            if Polygon(h.coords).area >= seuil_m2]
                    if len(kept) != len(list(p.interiors)):
                        p = Polygon(p.exterior.coords, kept)
                polys.append(p)
    return polys


# ── Conversion coordonnées ────────────────────────────────────────────────────
def lambert_to_mm(poly: Polygon, x_origin: float, y_origin: float,
                  echelle: int) -> Optional[Polygon]:
    """Lambert-93 (m) → SVG (mm). Y est inversé (nord = haut SVG = y=0)."""
    scale = 1000.0 / echelle

    def ring_to_mm(coords):
        return [(( x - x_origin) * scale,
                 (y_origin - y) * scale)
                for x, y in coords]

    try:
        ext  = ring_to_mm(poly.exterior.coords)
        ints = [ring_to_mm(r.coords) for r in poly.interiors]
        p = Polygon(ext, ints)
        return p if p.is_valid else p.buffer(0)
    except Exception:
        return None


# ── SVG ───────────────────────────────────────────────────────────────────────
def path_d(coords) -> str:
    pts = list(coords)
    if len(pts) < 2:
        return ""
    d = f"M {pts[0][0]:.3f},{pts[0][1]:.3f}"
    for x, y in pts[1:]:
        d += f" L {x:.3f},{y:.3f}"
    return d + " Z"


def path_d_poly(poly: Polygon) -> str:
    """Chemin SVG complet (extérieur + trous) avec fill-rule evenodd."""
    d = path_d(poly.exterior.coords)
    for ring in poly.interiors:
        d += " " + path_d(ring.coords)
    return d


def _poly_inset(poly: Polygon, retrait: float) -> Optional[Polygon]:
    """Retourne le polygone érodé de retrait mm, ou None si vide."""
    try:
        inner = poly.buffer(-retrait)
        if not inner.is_valid:
            inner = inner.buffer(0)
        return None if inner.is_empty else inner
    except Exception:
        return None


SHAPER_NS  = "http://www.shapertools.com/namespaces/shaper"
GRAVURE_MM = 0.5   # profondeur gravure légère (guide, label)


def _nouveau_dwg(fname: Path, w_mm: float, h_mm: float) -> svgwrite.Drawing:
    dwg = svgwrite.Drawing(
        str(fname),
        size=(f"{w_mm:.3f}mm", f"{h_mm:.3f}mm"),
        viewBox=f"0 0 {w_mm:.3f} {h_mm:.3f}",
        profile="full", debug=False
    )
    dwg.attribs["xmlns:shaper"] = SHAPER_NS
    return dwg


BBOX_EPS = 0.3   # mm — tolérance bords de carte (segments exclus du biseau V)


def _bevel_ring_paths(coords, w_mm: float, h_mm: float) -> List[str]:
    """
    Retourne les segments d'un anneau (extérieur ou trou) qui NE sont PAS
    sur les bords de la zone de carte. Seuls ces segments reçoivent le biseau V.
    """
    from shapely.geometry import LinearRing
    lr = LinearRing(list(coords))
    inset = shapely_box(BBOX_EPS, BBOX_EPS, w_mm - BBOX_EPS, h_mm - BBOX_EPS)
    segments = lr.intersection(inset)
    if segments.is_empty:
        return []
    geoms = list(segments.geoms) if hasattr(segments, 'geoms') else [segments]
    paths = []
    for seg in geoms:
        if not hasattr(seg, 'coords'):
            continue
        pts = list(seg.coords)
        if len(pts) < 2:
            continue
        d = f"M {pts[0][0]:.3f},{pts[0][1]:.3f}"
        for x, y in pts[1:]:
            d += f" L {x:.3f},{y:.3f}"
        paths.append(d)
    return paths


def generer_svg_couche(couche: Couche, polys_mm: List[Polygon],
                        w_mm: float, h_mm: float,
                        epaisseur: float, angle_fraise_v: float,
                        dossier: Path,
                        num_couche: int = 0,
                        polys_guide_mm: Optional[List[Polygon]] = None) -> int:
    """
    Génère un SVG par couche.
    MDF (face supérieure) : remplissage + guide N+1 + découpe + biseau V.
    Plexi (face inférieure, miroir) : même contenu mais en miroir X — la pièce
    est posée face-bas sur la Shaper, le biseau V crée le chanfrein côté marine.
    Le guide montre la couche marine juste en-dessous pour l'alignement.
    """
    is_plexi = couche.materiau == "plexi"
    ep_cut   = epaisseur          # même épaisseur plexi et MDF
    retrait  = epaisseur * math.tan(math.radians(angle_fraise_v))
    polys_valides = [p for p in polys_mm if p and not p.is_empty and p.is_valid]

    dwg = _nouveau_dwg(dossier / f"{num_couche:02d}_{couche.nom}.svg", w_mm, h_mm)

    # Pour le plexi : tout le contenu de découpe est dans un groupe miroir X.
    # Quand la pièce est retournée face-bas, la géométrie est correcte.
    if is_plexi:
        container = dwg.add(dwg.g(id="face_inferieure",
                                   transform=f"translate({w_mm:.3f},0) scale(-1,1)"))
    else:
        container = dwg

    # ── Remplissages visuels (sans shaper:cutDepth → affichage uniquement) ────
    g_fill_biseau = container.add(dwg.g(id="zone_biseau", stroke="none", fill="#C8C8C8",
                                        fill_rule="evenodd", opacity="1"))
    g_fill_int    = container.add(dwg.g(id="interieur",   stroke="none", fill="#F0F0F0",
                                        fill_rule="evenodd", opacity="1"))

    if is_plexi:
        # Les pièces plexi = pièces MER (opposé des terres) — remplissage direct
        for poly in polys_valides:
            g_fill_biseau.add(dwg.path(d=path_d_poly(poly)))
            inner = _poly_inset(poly, retrait)
            if inner:
                polys_inner = list(inner.geoms) if isinstance(inner, MultiPolygon) else [inner]
                for ip in polys_inner:
                    g_fill_int.add(dwg.path(d=path_d_poly(ip)))
    else:
        for poly in polys_valides:
            g_fill_biseau.add(dwg.path(d=path_d_poly(poly)))
            inner = _poly_inset(poly, retrait)
            if inner:
                polys_inner = list(inner.geoms) if isinstance(inner, MultiPolygon) else [inner]
                for ip in polys_inner:
                    g_fill_int.add(dwg.path(d=path_d_poly(ip)))

    # ── Guide — contours de référence en tirets orange ────────────────────────
    # MDF : guide = couche N+1 (au-dessus) ; Plexi : guide = couche marine N-1 (en-dessous)
    if polys_guide_mm:
        g_guide = container.add(dwg.g(id="guide", stroke="#FF8800", fill="none",
                                       stroke_width="0.15",
                                       **{"stroke-dasharray": "1,0.8",
                                          "shaper:cutDepth": f"{GRAVURE_MM}mm"}))
        for poly in [p for p in polys_guide_mm if p and not p.is_empty]:
            g_guide.add(dwg.path(d=path_d(poly.exterior.coords)))
            for ring in poly.interiors:
                g_guide.add(dwg.path(d=path_d(ring.coords)))

    # ── Découpe — fraise droite, pleine profondeur ────────────────────────────
    g_cut = container.add(dwg.g(id="decoupe", stroke="#FF0000", fill="none",
                                 stroke_width="0.1",
                                 **{"shaper:cutDepth": f"{ep_cut}mm"}))
    for poly in polys_valides:
        g_cut.add(dwg.path(d=path_d(poly.exterior.coords)))
        for ring in poly.interiors:
            g_cut.add(dwg.path(d=path_d(ring.coords)))

    # ── Biseau — fraise en V, segments hors bords de carte uniquement ─────────
    g_bevel = container.add(dwg.g(id="biseau", stroke="#0000FF", fill="none",
                                   stroke_width="0.1",
                                   **{"shaper:cutDepth": f"{retrait:.2f}mm"}))
    for poly in polys_valides:
        for d in _bevel_ring_paths(poly.exterior.coords, w_mm, h_mm):
            g_bevel.add(dwg.path(d=d))
        for ring in poly.interiors:
            for d in _bevel_ring_paths(ring.coords, w_mm, h_mm):
                g_bevel.add(dwg.path(d=d))

    # ── Numéro de couche gravé — repère de montage ────────────────────────────
    if num_couche > 0:
        label = f"{num_couche:02d}"
        g_num = container.add(dwg.g(id="numero_couche", stroke="none", fill="#444444",
                                     **{"shaper:cutDepth": f"{GRAVURE_MM}mm"}))
        for poly in polys_valides:
            cx, cy = poly.centroid.x, poly.centroid.y
            if is_plexi:
                # Contre-miroir du texte pour qu'il soit lisible face-bas
                g_num.add(dwg.g(
                    transform=f"translate({cx:.2f},{cy:.2f}) scale(-1,1)"
                ).add(dwg.text(label, insert=(0, 0),
                               font_size="5mm", font_family="sans-serif",
                               text_anchor="middle", dominant_baseline="middle")))
            else:
                g_num.add(dwg.text(label, insert=(f"{cx:.2f}", f"{cy:.2f}"),
                                   font_size="5mm", font_family="sans-serif",
                                   text_anchor="middle", dominant_baseline="middle"))

    dwg.save(pretty=True)
    return len(polys_valides)


# ── Guide de montage ──────────────────────────────────────────────────────────
def generer_guide_montage(couches: List[Couche], all_polys_mm: List[List[Polygon]],
                           w_mm: float, h_mm: float, dossier: Path):
    """
    Génère 00_guide_montage.svg : grille A4 paysage de vignettes, une par couche.
    Ordre bas → haut : couche 01 en haut-gauche, 20 en bas-droite.
    """
    PAGE_W, PAGE_H = 297.0, 210.0   # A4 paysage (mm)
    N_COLS, N_ROWS = 5, 4           # 5×4 = 20 vignettes
    MARGE_PAGE = 5.0
    TITLE_H    = 12.0
    LABEL_H    = 9.0                # hauteur zone texte sous chaque vignette
    PAD        = 2.0                # padding interne entre vignettes

    cell_w = (PAGE_W - 2 * MARGE_PAGE) / N_COLS
    cell_h = (PAGE_H - TITLE_H - 2 * MARGE_PAGE) / N_ROWS

    # Dimensions de la vignette (respecte le rapport de la carte)
    aspect = w_mm / h_mm
    thumb_w = cell_w - 2 * PAD
    thumb_h = thumb_w / aspect
    if thumb_h > cell_h - LABEL_H - 2 * PAD:
        thumb_h = cell_h - LABEL_H - 2 * PAD
        thumb_w = thumb_h * aspect
    scale = thumb_w / w_mm

    dwg = svgwrite.Drawing(
        str(dossier / "00_guide_montage.svg"),
        size=(f"{PAGE_W}mm", f"{PAGE_H}mm"),
        viewBox=f"0 0 {PAGE_W} {PAGE_H}",
        profile="full", debug=False
    )

    # Fond blanc
    dwg.add(dwg.rect(insert=(0, 0), size=(f"{PAGE_W}", f"{PAGE_H}"), fill="white"))

    # Titre
    dwg.add(dwg.text("GUIDE DE MONTAGE — Port-Blanc (Penvénan)",
                     insert=(f"{MARGE_PAGE}", "8"),
                     font_size="5mm", font_family="sans-serif", font_weight="bold",
                     fill="#111111"))
    n_last = len(couches) - 1  # plexi + terre partagent un numéro → total - 1
    dwg.add(dwg.text(f"Assemblage de bas en haut : couche 01 (marine profonde) → {n_last:02d} (sommet)",
                     insert=(f"{PAGE_W / 2}", "8"),
                     font_size="2.8mm", font_family="sans-serif", fill="#555555",
                     text_anchor="middle"))

    plexi_idx = next((j for j, c in enumerate(couches) if c.materiau == "plexi"), -1)

    for i, (couche, polys_mm) in enumerate(zip(couches, all_polys_mm)):
        col = i % N_COLS
        row = i // N_COLS
        num = i + 1 if i <= plexi_idx else i

        color   = couche.couleur if couche.materiau != "plexi" else "#89c4e1"
        polys_v = [p for p in polys_mm if p and not p.is_empty]

        # Origine de la cellule
        cell_x = MARGE_PAGE + col * cell_w
        cell_y = TITLE_H + MARGE_PAGE + row * cell_h

        # Vignette centrée dans la cellule
        tx = cell_x + (cell_w - thumb_w) / 2
        ty = cell_y + PAD

        # ── Clip path ─────────────────────────────────────────────────────────
        clip_id = f"clip{num:02d}"
        clip = dwg.defs.add(dwg.clipPath(id=clip_id))
        clip.add(dwg.rect(insert=(f"{tx:.2f}", f"{ty:.2f}"),
                          size=(f"{thumb_w:.2f}", f"{thumb_h:.2f}")))

        # ── Fond vignette ─────────────────────────────────────────────────────
        dwg.add(dwg.rect(insert=(f"{tx:.2f}", f"{ty:.2f}"),
                         size=(f"{thumb_w:.2f}", f"{thumb_h:.2f}"),
                         fill="#f0f0f0", stroke="#aaaaaa", stroke_width="0.2"))

        # ── Formes de la couche ───────────────────────────────────────────────
        def scaled_path(coords):
            pts = [(x * scale + tx, y * scale + ty) for x, y in coords]
            return path_d(pts)

        g_fill = dwg.add(dwg.g(clip_path=f"url(#{clip_id})"))

        if couche.materiau == "plexi":
            # Mer = gris coloré, terre = trou (evenodd)
            d = (f"M {tx:.2f},{ty:.2f} L {tx+thumb_w:.2f},{ty:.2f} "
                 f"L {tx+thumb_w:.2f},{ty+thumb_h:.2f} L {tx:.2f},{ty+thumb_h:.2f} Z")
            for poly in polys_v:
                d += " " + scaled_path(poly.exterior.coords)
            g_fill.add(dwg.path(d=d, fill=color, stroke="none",
                                fill_rule="evenodd", opacity="0.75"))
        else:
            for poly in polys_v:
                g_fill.add(dwg.path(d=scaled_path(poly.exterior.coords),
                                    fill=color, stroke="none", opacity="0.75"))

        # ── Guide couche suivante (tirets fins) ───────────────────────────────
        if i + 1 < len(all_polys_mm):
            g_guide = dwg.add(dwg.g(clip_path=f"url(#{clip_id})",
                                    stroke="#FF8800", fill="none",
                                    stroke_width="0.2",
                                    **{"stroke-dasharray": "0.6,0.4"}))
            for poly in [p for p in all_polys_mm[i + 1] if p and not p.is_empty]:
                g_guide.add(dwg.path(d=scaled_path(poly.exterior.coords)))

        # ── Numéro (grand, coin haut-gauche) ─────────────────────────────────
        dwg.add(dwg.text(f"{num:02d}",
                         insert=(f"{tx + 1.5:.2f}", f"{ty + 5.5:.2f}"),
                         font_size="5.5mm", font_family="sans-serif", font_weight="bold",
                         fill="white", stroke="#333333", stroke_width="0.25",
                         clip_path=f"url(#{clip_id})"))

        # ── Texte sous la vignette ────────────────────────────────────────────
        label_y = ty + thumb_h + 3.5
        n_p = len(polys_v)
        if couche.materiau == "plexi":
            alt = "z = 0 m"
            mat = "PLEXIGLASS"
        else:
            alt = f"{couche.z_low:+.0f}→{couche.z_high:+.0f}m"
            mat = "MDF"
        dwg.add(dwg.text(f"{alt}  ·  {mat}  ·  {n_p} pce{'s' if n_p > 1 else ''}",
                         insert=(f"{tx + thumb_w / 2:.2f}", f"{label_y:.2f}"),
                         font_size="2.3mm", font_family="sans-serif",
                         fill="#333333", text_anchor="middle"))

    dwg.save(pretty=True)
    print(f"  Guide de montage   → {dossier}/00_guide_montage.svg")


# ── make_args ─────────────────────────────────────────────────────────────────
def make_args(**kwargs) -> types.SimpleNamespace:
    """Construit un objet args compatible avec les fonctions run_*."""
    defaults = dict(
        mnt=MNT_DEFAUT, lat=48.841, lon=-3.302,
        largeur=4.3, hauteur=3.8,
        echelle=10000, equidistance=5,
        epaisseur=3.0, seuil_surface=2000.0,
        angle_fraise_v=45.0, fraise_mm=3.0,
        simplification=None, lissage=None, methode_lissage="spline",
        dossier_sortie=SORTIE_DEFAUT,
        preview=False, generate=False, analyse=False,
        png=False, show_pieces=False,
    )
    defaults.update(kwargs)
    a = types.SimpleNamespace(**defaults)
    fraise_reel_m = a.fraise_mm * a.echelle / 1000.0
    if a.simplification is None:
        a.simplification = fraise_reel_m
    if a.lissage is None:
        a.lissage = fraise_reel_m / 2.0
    return a


# ── Analyse des seuils ────────────────────────────────────────────────────────
SEUILS_ANALYSE = [500, 1000, 2000, 5000, 10000, 20000, 50000]


def run_analyse_data(args, data: np.ma.MaskedArray, raster_transform, bounds,
                     couches: List[Couche], progress_cb=None) -> dict:
    """Retourne les données d'analyse sous forme de dict (pour l'API web)."""
    seuils   = SEUILS_ANALYSE
    scale_mm = 1000.0 / args.echelle
    result_couches, totaux = [], [0] * len(seuils)

    for idx, couche in enumerate(couches):
        if couche.materiau == "plexi":
            continue
        polys = extraire_polygones(data, couche.z_low, raster_transform, 0,
                                   args.simplification, args.lissage,
                                   getattr(args, 'methode_lissage', 'spline'))
        areas  = [p.area for p in polys]
        counts = [sum(1 for a in areas if a >= s) for s in seuils]
        for i, c in enumerate(counts):
            totaux[i] += c
        result_couches.append(dict(
            nom=couche.nom, z_low=couche.z_low, z_high=couche.z_high,
            materiau=couche.materiau, couleur=couche.couleur, counts=counts,
        ))
        if progress_cb:
            progress_cb(len(result_couches))

    correspondances = [
        dict(seuil=s,
             mm2=round(s * scale_mm ** 2, 2),
             cote=round(math.sqrt(s * scale_mm ** 2), 2))
        for s in seuils
    ]
    return dict(seuils=seuils, couches=result_couches,
                totaux=totaux, correspondances=correspondances)


def run_analyse(args, data: np.ma.MaskedArray, raster_transform, bounds, couches: List[Couche]):
    echelle = args.echelle
    print(f"\nAnalyse des seuils de surface minimale")
    print(f"  Simplification: {args.simplification} m  |  Lissage: {args.lissage} m")
    print(f"  Équidistance  : {args.equidistance} m\n")

    r     = run_analyse_data(args, data, raster_transform, bounds, couches)
    col_w = 10
    header = f"{'Couche':<30}" + "".join(f"{s:>{col_w}}" for s in r['seuils'])
    print(header)
    print("-" * len(header))
    for c in r['couches']:
        print(f"{c['nom']:<30}" + "".join(f"{v:>{col_w}}" for v in c['counts']))
    print("-" * len(header))
    print(f"{'TOTAL':<30}" + "".join(f"{t:>{col_w}}" for t in r['totaux']))
    print(f"\n  Correspondance seuil → taille sur la carte (échelle 1:{echelle}) :")
    print(f"  {'Seuil (m²)':>12}  {'Surface carte (mm²)':>22}  {'≈ côté carré (mm)':>20}")
    for c in r['correspondances']:
        print(f"  {c['seuil']:>12}  {c['mm2']:>22.2f}  {c['cote']:>18.2f}")


# ── Preview ───────────────────────────────────────────────────────────────────
def run_preview(args, data: np.ma.MaskedArray, raster_transform, bounds, couches: List[Couche]):
    equi = args.equidistance

    # Colormap par couche
    cmap_colors, cmap_bounds = [], []
    for c in couches:
        if c.materiau == "plexi":
            continue
        cmap_colors.append(c.couleur)
        cmap_bounds.append(c.z_low)
    cmap_bounds.append(couches[-1].z_high)

    cmap = mcolors.ListedColormap(cmap_colors)
    norm = mcolors.BoundaryNorm(cmap_bounds, cmap.N)

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor('#1e1e2e')

    ax     = fig.add_axes([0.04, 0.10, 0.63, 0.84])
    ax_btn = fig.add_axes([0.27, 0.02, 0.26, 0.045])
    ax_leg = fig.add_axes([0.69, 0.10, 0.30, 0.84])
    ax.set_facecolor('#0d1b2a')
    ax_leg.set_facecolor('#1e1e2e')
    ax_leg.axis('off')

    ext = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    if args.show_pieces:
        # Colorie chaque pièce individuellement
        rng = np.random.default_rng(42)
        display = np.zeros((*data.shape, 4), dtype=float)
        for c in couches:
            if c.materiau == "plexi":
                continue
            polys = extraire_polygones(data, c.z_low, raster_transform, args.seuil_surface)
            for poly in polys:
                color = rng.uniform(0.2, 0.95, 3)
                # Rasteriser chaque polygone (approche simplifiée : bbox)
                # Pour un vrai rendu pièce par pièce on utiliserait rasterio.features.rasterize
                pass  # aperçu basique
        # Fallback : affichage normal
        im = ax.imshow(data, cmap=cmap, norm=norm, extent=ext,
                       origin='upper', aspect='equal', interpolation='none')
        ax.set_title("Preview — mode pièces (contours)", fontsize=10, color='white')
    else:
        im = ax.imshow(data, cmap=cmap, norm=norm, extent=ext,
                       origin='upper', aspect='equal', interpolation='none')

    # Courbes de niveau
    z_min = float(data.min())
    z_max = float(data.max())
    levels = np.arange(math.floor(z_min / equi) * equi,
                       math.ceil(z_max / equi) * equi + equi, equi)
    X = np.linspace(bounds.left, bounds.right, data.shape[1])
    Y = np.linspace(bounds.top, bounds.bottom, data.shape[0])
    try:
        ax.contour(X, Y, data.filled(np.nan), levels=levels,
                   colors='white', linewidths=0.35, alpha=0.55)
    except Exception:
        pass

    # Couche plexiglass semi-transparente
    plexi = mpatches.Rectangle(
        (bounds.left, bounds.bottom),
        bounds.right - bounds.left, bounds.top - bounds.bottom,
        facecolor='white', alpha=0.22, zorder=5, label='Plexi visible'
    )
    ax.add_patch(plexi)

    # Cadre carte
    ax.add_patch(mpatches.Rectangle(
        (bounds.left, bounds.bottom),
        bounds.right - bounds.left, bounds.top - bounds.bottom,
        linewidth=1.5, edgecolor='#f38ba8', fill=False, zorder=6
    ))

    # Trous goupilles (indicatifs)
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        gx = bounds.left + dx * (bounds.right - bounds.left)
        gy = bounds.bottom + dy * (bounds.top - bounds.bottom)
        ax.plot(gx, gy, '+', color='#a6e3a1', markersize=8, markeredgewidth=1.2, zorder=7)

    ax.set_title(f"Preview Carte 3D — Port-Blanc  |  équidistance {equi} m",
                 fontsize=11, color='white', pad=8)
    ax.tick_params(colors='#6c7086', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#45475a')

    # Légende
    handles = []
    for c in couches:
        if c.materiau == "plexi":
            handles.append(mpatches.Patch(facecolor='#d0eeff', alpha=0.5,
                                          edgecolor='#89b4fa', linewidth=0.8,
                                          label="0 m — Plexiglass"))
        else:
            handles.append(mpatches.Patch(facecolor=c.couleur,
                                          label=f"{c.z_low:+.0f} → {c.z_high:+.0f} m"))
    ax_leg.legend(handles=handles, loc='upper left', fontsize=7.5,
                  framealpha=0.15, labelcolor='#cdd6f4',
                  title="Couches", title_fontsize=8,
                  facecolor='#313244', edgecolor='#45475a')

    # Résumé
    n_marine = sum(1 for c in couches if c.materiau == "mdf" and c.z_low < 0)
    n_terre  = sum(1 for c in couches if c.materiau == "mdf" and c.z_low >= 0)
    w_mm = (bounds.right - bounds.left) * 1000 / args.echelle
    h_mm = (bounds.top - bounds.bottom) * 1000 / args.echelle
    retrait = args.epaisseur * math.tan(math.radians(args.angle_fraise_v))
    info = (f"Format : {w_mm:.0f} × {h_mm:.0f} mm  |  "
            f"{n_marine} marine + 1 plexi + {n_terre} terre = {len(couches)} couches\n"
            f"Alt : {z_min:.1f} → {z_max:.1f} m  |  "
            f"Fraise V {args.angle_fraise_v}° → retrait {retrait:.2f} mm")
    fig.text(0.04, 0.005, info, fontsize=7.5, color='#6c7086',
             verticalalignment='bottom')

    # Bouton toggle plexiglass
    btn = Button(ax_btn, '👁  Niveau mer (plexiglass)',
                 color='#313244', hovercolor='#45475a')
    btn.label.set_color('#cdd6f4')
    btn.label.set_fontsize(9)

    def toggle_plexi(event):
        plexi.set_visible(not plexi.get_visible())
        btn.label.set_text('👁  Niveau mer (visible)' if plexi.get_visible()
                           else '🚫  Niveau mer (masqué)')
        fig.canvas.draw()
    btn.on_clicked(toggle_plexi)

    if args.png:
        out = Path(args.dossier_sortie) / "preview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Preview exportée → {out}")
    else:
        plt.show()


# ── Génération SVG ────────────────────────────────────────────────────────────
def run_generate(args, data: np.ma.MaskedArray, raster_transform, bounds, couches: List[Couche],
                 progress_cb=None):
    echelle  = args.echelle
    ep       = args.epaisseur
    seuil    = args.seuil_surface
    angle_v  = args.angle_fraise_v
    retrait  = ep * math.tan(math.radians(angle_v))   # angle_v = angle de flanc

    x_origin = bounds.left
    y_origin = bounds.top
    w_mm = (bounds.right - bounds.left) * 1000.0 / echelle
    h_mm = (bounds.top - bounds.bottom) * 1000.0 / echelle

    dossier = Path(args.dossier_sortie)
    dossier.mkdir(parents=True, exist_ok=True)

    print(f"\nGénération SVG")
    print(f"  Format        : {w_mm:.1f} × {h_mm:.1f} mm")
    print(f"  Retrait V     : {retrait:.2f} mm  (fraise {angle_v}°, ep. {ep} mm)")
    print(f"  Seuil         : {seuil:.0f} m²  ({seuil / 1e6:.4f} km²)")
    print(f"  Simplification: {args.simplification} m réels  →  {args.simplification * 1000 / echelle:.2f} mm carte")
    print(f"  Lissage       : {args.lissage} m réels  →  {args.lissage * 1000 / echelle:.2f} mm carte  ({args.methode_lissage})")
    print(f"  Retrait biseau: {retrait:.2f} mm  (flanc {angle_v}°, ep. {ep} mm  →  {ep}×tan({angle_v}°))")
    print(f"  Dossier       : {dossier}/\n")

    total_pieces = 0

    def calc_polys_mm(couche: Couche) -> List[Polygon]:
        if couche.materiau == "plexi":
            # Plexi = exact opposé de la couche terre z=0 :
            # pièces = zone MER (cadre entier moins les polygones terres)
            terre_polys = extraire_polygones(data, 0.0, raster_transform, seuil,
                                             args.simplification, args.lissage,
                                             args.methode_lissage)
            terre_mm = [lambert_to_mm(p, x_origin, y_origin, echelle) for p in terre_polys]
            terre_mm = [p for p in terre_mm if p is not None and not p.is_empty]
            cadre = shapely_box(0, 0, w_mm, h_mm)
            from shapely.ops import unary_union
            if terre_mm:
                union_terre = unary_union(terre_mm)
                mer = cadre.difference(union_terre)
            else:
                mer = cadre
            if mer.is_empty:
                return []
            geoms = list(mer.geoms) if isinstance(mer, MultiPolygon) else [mer]
            return [g for g in geoms if isinstance(g, Polygon) and g.area >= seuil * (1000 / echelle) ** 2]
        polys_l93 = extraire_polygones(data, couche.z_low, raster_transform, seuil,
                                       args.simplification, args.lissage,
                                       args.methode_lissage)
        result = [lambert_to_mm(p, x_origin, y_origin, echelle) for p in polys_l93]
        return [p for p in result if p is not None and not p.is_empty]

    # Calcul de tous les polygones (nécessaire pour le guide de montage)
    print("  Extraction des polygones...", end="", flush=True)
    all_polys_mm = [calc_polys_mm(c) for c in couches]
    print("  OK")

    from shapely.ops import unary_union

    def _clip_polys(polys, envelope):
        """Intersecte une liste de polygones avec une enveloppe, retourne des Polygon."""
        result = []
        for p in polys:
            inter = p.intersection(envelope)
            if inter.is_empty:
                continue
            if hasattr(inter, 'geoms'):
                result.extend(g for g in inter.geoms
                              if isinstance(g, Polygon) and not g.is_empty)
            elif isinstance(inter, Polygon):
                result.append(inter)
        return result

    def _ref_layer(i):
        """Couche de référence pour l'emboîtement : saute le plexi."""
        if couches[i].materiau == "plexi":
            return -1                       # ne pas clipper le plexi lui-même
        ref = i - 1
        if ref >= 0 and couches[ref].materiau == "plexi":
            ref -= 1                        # terre_+0_+5 → référence = marine_-5_+0
        return ref

    # ── 1. Emboîtement : couche N+1 ⊂ couche N (cascadé bas → haut) ─────
    #    Le lissage indépendant casse l'emboîtement naturel du raster.
    #    On force couche[i] = couche[i] ∩ couche[ref] pour le restaurer.
    n_nested = 0
    for i in range(1, len(couches)):
        ref = _ref_layer(i)
        if ref < 0 or not all_polys_mm[ref] or not all_polys_mm[i]:
            continue
        enveloppe = unary_union(all_polys_mm[ref])
        before = len(all_polys_mm[i])
        all_polys_mm[i] = _clip_polys(all_polys_mm[i], enveloppe)
        if len(all_polys_mm[i]) != before:
            n_nested += 1
    if n_nested:
        print(f"  Emboîtement forcé : {n_nested} couche(s) ajustée(s)")

    # ── 2. Clipping chanfrein : couche N+1 ⊂ inset(couche N, retrait) ────
    #    Garantit que la couche supérieure ne déborde pas sur la zone biseautée.
    n_clipped = 0
    for i in range(1, len(couches)):
        ref = _ref_layer(i)
        if ref < 0 or not all_polys_mm[ref] or not all_polys_mm[i]:
            continue
        enveloppe_inf = unary_union(all_polys_mm[ref]).buffer(-retrait)
        if enveloppe_inf.is_empty:
            continue
        before = len(all_polys_mm[i])
        all_polys_mm[i] = _clip_polys(all_polys_mm[i], enveloppe_inf)
        if len(all_polys_mm[i]) != before:
            n_clipped += 1
    if n_clipped:
        print(f"  Clipping chanfrein : {n_clipped} couche(s) ajustée(s)")

    # Aplatir tout MultiPolygon résiduel en Polygon individuels
    for i in range(len(all_polys_mm)):
        flat = []
        for p in all_polys_mm[i]:
            if isinstance(p, MultiPolygon):
                flat.extend(g for g in p.geoms if isinstance(g, Polygon) and not g.is_empty)
            elif isinstance(p, Polygon) and not p.is_empty:
                flat.append(p)
        all_polys_mm[i] = flat

    # ── Suppression des trous cachés ─────────────────────────────────────────
    #    Un trou dans la couche i est inutile s'il est entièrement couvert par
    #    la couche i+1 (qui sera posée par-dessus). Ça simplifie la découpe.
    n_holes_removed = 0
    for i in range(len(couches) - 1):
        if couches[i].materiau == "plexi" or couches[i + 1].materiau == "plexi":
            continue
        if not all_polys_mm[i] or not all_polys_mm[i + 1]:
            continue
        couche_sup = unary_union(all_polys_mm[i + 1])
        cleaned = []
        for p in all_polys_mm[i]:
            if not p.interiors:
                cleaned.append(p)
                continue
            visible_holes = []
            for hole in p.interiors:
                hole_poly = Polygon(hole.coords)
                if not couche_sup.contains(hole_poly):
                    visible_holes.append(hole)
                else:
                    n_holes_removed += 1
            cleaned.append(Polygon(p.exterior.coords, visible_holes))
        all_polys_mm[i] = cleaned
    if n_holes_removed:
        print(f"  Trous cachés supprimés : {n_holes_removed}")
    print()

    # Numérotation : plexi et première couche terre partagent le même niveau
    plexi_idx = next((j for j, c in enumerate(couches) if c.materiau == "plexi"), -1)

    for i, couche in enumerate(couches):
        num = i + 1 if i <= plexi_idx else i
        print(f"  [{couche.nom}]  z={couche.z_low:+.0f}m  ", end="", flush=True)
        # Guide MDF : couche N+1 (au-dessus) ; Plexi : couche marine N-1 (en-dessous)
        if couche.materiau == "plexi":
            polys_guide = all_polys_mm[i - 1] if i > 0 else []
        else:
            polys_guide = all_polys_mm[i + 1] if i + 1 < len(couches) else []
        n = generer_svg_couche(couche, all_polys_mm[i], w_mm, h_mm, ep, angle_v, dossier,
                               num_couche=num,
                               polys_guide_mm=polys_guide)
        total_pieces += n
        if progress_cb:
            progress_cb(i + 1, len(couches), couche.nom, n)
        print(f"{n} pièces")

    print(f"\n  Total : {total_pieces} pièces  →  {dossier}/")
    generer_guide_montage(couches, all_polys_mm, w_mm, h_mm, dossier)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="carte3d.py — Génération SVG pour carte topo-bathymétrique 3D")
    p.add_argument("--mnt",             default=MNT_DEFAUT)
    p.add_argument("--lat",             type=float, default=48.841)
    p.add_argument("--lon",             type=float, default=-3.302)
    p.add_argument("--largeur",         type=float, default=4.3)
    p.add_argument("--hauteur",         type=float, default=3.8)
    p.add_argument("--echelle",         type=int,   default=10000)
    p.add_argument("--equidistance",    type=int,   default=5)
    p.add_argument("--epaisseur",       type=float, default=3.0)
    p.add_argument("--seuil_surface",   type=float, default=1000.0,
                   help="Surface min d'une pièce en m² (défaut: 1000 = 0.1ha)")
    p.add_argument("--angle_fraise_v",  type=float, default=45.0,
                   help="Angle de flanc de la fraise en V en degrés (défaut: 45 → angle inclus 90°). "
                        "Retrait surface = épaisseur × tan(angle_flanc)")
    p.add_argument("--simplification",  type=float, default=None,
                   help="Tolérance Douglas-Peucker en mètres réels (défaut: calculé depuis --fraise_mm)")
    p.add_argument("--lissage",         type=float, default=None,
                   help="Rayon de lissage par double-buffer en mètres réels (défaut: calculé depuis --fraise_mm, 0=désactivé)")
    p.add_argument("--methode_lissage",  default="spline", choices=["spline", "buffer"],
                   help="Méthode de lissage: spline (B-spline, courbes organiques, défaut) "
                        "ou buffer (double-buffer, méthode legacy)")
    p.add_argument("--fraise_mm",       type=float, default=3.0,
                   help="Diamètre de la plus petite fraise droite en mm (défaut: 3). "
                        "Détermine automatiquement simplification et lissage.")
    p.add_argument("--dossier_sortie",  default=SORTIE_DEFAUT)
    p.add_argument("--preview",         action="store_true")
    p.add_argument("--generate",        action="store_true")
    p.add_argument("--analyse",         action="store_true",
                   help="Affiche le nombre de pièces par couche selon différents seuils")
    p.add_argument("--png",             action="store_true",
                   help="Exporter la preview en PNG (sans affichage)")
    p.add_argument("--show-pieces",     dest="show_pieces", action="store_true",
                   help="Colorier chaque pièce individuellement")
    args = p.parse_args()

    if not args.preview and not args.generate and not args.analyse:
        p.print_help()
        return

    # Résolution minimale dérivée du diamètre de fraise
    # fraise_mm sur la carte → fraise_mm * echelle / 1000 mètres réels
    fraise_reel_m = args.fraise_mm * args.echelle / 1000.0
    if args.simplification is None:
        args.simplification = fraise_reel_m          # = diamètre fraise en réel
    if args.lissage is None:
        args.lissage = fraise_reel_m / 2.0           # = rayon fraise en réel

    # Chargement MNT
    print(f"Chargement MNT : {args.mnt}")
    with rasterio.open(args.mnt) as src:
        data_raw        = src.read(1).astype(float)
        nodata          = src.nodata
        raster_transform = src.transform
        bounds          = src.bounds

    if nodata is not None:
        data = np.ma.masked_where(data_raw == nodata, data_raw)
    else:
        data = np.ma.array(data_raw)

    valid = data.compressed()
    z_min, z_max = float(valid.min()), float(valid.max())
    print(f"  Dimensions : {data.shape[1]} × {data.shape[0]} px  |  Alt : {z_min:.1f} → {z_max:.1f} m")

    couches = definir_couches(z_min, z_max, args.equidistance)
    n_m = sum(1 for c in couches if c.materiau == "mdf" and c.z_low < 0)
    n_t = sum(1 for c in couches if c.materiau == "mdf" and c.z_low >= 0)
    print(f"  Couches    : {n_m} marine + 1 plexi + {n_t} terre = {len(couches)} total")

    if args.analyse:
        run_analyse(args, data, raster_transform, bounds, couches)

    if args.preview:
        run_preview(args, data, raster_transform, bounds, couches)

    if args.generate:
        run_generate(args, data, raster_transform, bounds, couches)


if __name__ == "__main__":
    main()
