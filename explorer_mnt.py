#!/usr/bin/env python3
"""
explorer_mnt.py — Exploration d'un fichier MNT (ASC ou GeoTIFF)
Usage : python explorer_mnt.py <fichier.asc|.tif>
"""

import sys
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def explorer(chemin: str):
    path = Path(chemin)
    if not path.exists():
        print(f"Erreur : fichier introuvable → {chemin}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Fichier : {path.name}")
    print(f"{'='*60}\n")

    with rasterio.open(path) as src:
        # ── Métadonnées ──────────────────────────────────────────────
        print(f"[Projection]")
        print(f"  CRS       : {src.crs}")
        print(f"  EPSG      : {src.crs.to_epsg() if src.crs else '?'}")
        print()

        print(f"[Dimensions]")
        print(f"  Colonnes  : {src.width} px")
        print(f"  Lignes    : {src.height} px")
        print(f"  Bandes    : {src.count}")
        print(f"  Résolution: {src.res[0]:.2f} × {src.res[1]:.2f} m/px")
        print()

        # ── Emprise en coordonnées natives ───────────────────────────
        b = src.bounds
        print(f"[Emprise native ({src.crs.to_epsg() if src.crs else 'CRS inconnu'})]")
        print(f"  Xmin (W)  : {b.left:.2f}")
        print(f"  Xmax (E)  : {b.right:.2f}")
        print(f"  Ymin (S)  : {b.bottom:.2f}")
        print(f"  Ymax (N)  : {b.top:.2f}")
        print(f"  Largeur   : {(b.right - b.left) / 1000:.2f} km")
        print(f"  Hauteur   : {(b.top - b.bottom) / 1000:.2f} km")
        print()

        # ── Emprise en WGS84 ─────────────────────────────────────────
        # Les fichiers ASC Litto3D n'embarquent pas le CRS → on assume Lambert-93
        crs_source = src.crs if src.crs else CRS.from_epsg(2154)
        if not src.crs:
            print(f"  (CRS absent du fichier — Lambert-93 / EPSG:2154 assumé)\n")
        try:
            wgs84 = CRS.from_epsg(4326)
            w, s, e, n = transform_bounds(crs_source, wgs84, b.left, b.bottom, b.right, b.top)
            print(f"[Emprise WGS84]")
            print(f"  N : {n:.5f}°")
            print(f"  S : {s:.5f}°")
            print(f"  E : {e:.5f}°")
            print(f"  W : {w:.5f}°")
            print(f"  Centre : {(n+s)/2:.5f}°N, {(e+w)/2:.5f}°E")
            print()
        except Exception as ex:
            print(f"  (Conversion WGS84 échouée : {ex})\n")

        # ── Lecture des données ──────────────────────────────────────
        print("Lecture des données... ", end="", flush=True)
        data = src.read(1, masked=True)
        nodata = src.nodata
        print("OK")
        print()

        valid = data.compressed()  # valeurs sans nodata

        print(f"[Altitudes / profondeurs]")
        print(f"  No-data   : {nodata}")
        print(f"  Pixels valides : {len(valid):,} / {data.size:,} ({100*len(valid)/data.size:.1f}%)")
        print(f"  Min       : {valid.min():.2f} m")
        print(f"  Max       : {valid.max():.2f} m")
        print(f"  Moyenne   : {valid.mean():.2f} m")
        print(f"  Médiane   : {np.median(valid):.2f} m")
        print()

        # ── Distribution par tranches ────────────────────────────────
        print(f"[Distribution par tranches de 5 m]")
        tranches = [(-50, -20), (-20, -15), (-15, -10), (-10, -5), (-5, 0),
                    (0, 10), (10, 20), (20, 30), (30, 50), (50, 200)]
        for lo, hi in tranches:
            mask = (valid >= lo) & (valid < hi)
            n_px = mask.sum()
            pct = 100 * n_px / len(valid)
            bar = '█' * int(pct / 2)
            print(f"  [{lo:>5} → {hi:>4} m] {bar:<25} {pct:5.1f}%  ({n_px:,} px)")
        print()

        # ── Visualisation ────────────────────────────────────────────
        print("Génération de la visualisation...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Exploration MNT — {path.name}", fontsize=13, fontweight='bold')

        vmin, vmax = max(valid.min(), -20), min(valid.max(), 60)

        # Carte de couleur : bleu (marine) → vert (terre)
        # Les points sont normalisés 0→1 sur la plage [vmin, vmax]
        zero_norm = max(0.01, min(0.99, (0 - vmin) / (vmax - vmin)))
        cmap = mcolors.LinearSegmentedColormap.from_list('topo', [
            (0.0,                           '#0d2b52'),  # bleu nuit profond (min)
            (max(0.01, zero_norm * 0.5),    '#2E6DA4'),  # bleu moyen
            (max(0.02, zero_norm * 0.9),    '#89c4e1'),  # bleu clair (estran)
            (zero_norm,                     '#C8A96E'),  # sable (zéro)
            (min(zero_norm + 0.10, 0.97),   '#5D8A3C'),  # vert clair
            (min(zero_norm + 0.35, 0.98),   '#3B5E2B'),  # vert foncé
            (1.0,                           '#8B7355'),  # brun (sommets)
        ], N=256)

        # Vue carte
        ax = axes[0]
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='bilinear', aspect='equal')
        ax.contour(data, levels=np.arange(np.floor(vmin/5)*5, vmax, 5),
                   colors='white', linewidths=0.3, alpha=0.5)
        ax.axhline(y=np.argmin(np.abs(np.linspace(vmax, vmin, data.shape[0]))),
                   color='cyan', linewidth=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Altitude (m)', fraction=0.03)
        ax.set_title("Vue d'ensemble")
        ax.set_xlabel("Colonnes (px)")
        ax.set_ylabel("Lignes (px)")

        # Histogramme
        ax2 = axes[1]
        bins = np.arange(np.floor(valid.min()/5)*5, np.ceil(valid.max()/5)*5 + 5, 5)
        n_hist, edges, patches = ax2.hist(valid, bins=bins, color='#89b4fa',
                                           edgecolor='#313244', linewidth=0.3)
        # Colorier les barres selon l'altitude
        for patch, left in zip(patches, edges[:-1]):
            if left < 0:
                patch.set_facecolor('#2E6DA4')
            elif left < 10:
                patch.set_facecolor('#C8A96E')
            elif left < 25:
                patch.set_facecolor('#5D8A3C')
            else:
                patch.set_facecolor('#3B5E2B')
        ax2.axvline(x=0, color='cyan', linewidth=1.5, linestyle='--', label='Niveau mer (0 m)')
        ax2.set_xlabel("Altitude (m)")
        ax2.set_ylabel("Nombre de pixels")
        ax2.set_title("Distribution des altitudes")
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("Terminé.\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage : python explorer_mnt.py <fichier.asc|.tif>")
        print("Exemple : python explorer_mnt.py LITTO3D_FXX_0277_6830_MNT_LAMB93_IGN69.asc")
        sys.exit(0)
    explorer(sys.argv[1])
