#!/usr/bin/env python3
"""
fusionner_mnt.py — Fusionne toutes les dalles Litto3D MNT1m en un seul GeoTIFF
                   et recadre sur la zone d'intérêt GPS.

Usage :
  python fusionner_mnt.py                          # zone par défaut (Port-Blanc)
  python fusionner_mnt.py --lat 48.841 --lon -3.302 --largeur 4.3 --hauteur 3.8
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds
from shapely.geometry import box
import json


# ── Paramètres par défaut (Port-Blanc) ────────────────────────────────────────
DEFAULT_LAT     = 48.841
DEFAULT_LON     = -3.302
DEFAULT_LARGEUR = 4.3   # km
DEFAULT_HAUTEUR = 3.8   # km
DATA_DIR        = Path("data")
SORTIE          = Path("data/mnt_fusion.tif")

CRS_LAMB93 = CRS.from_epsg(2154)
CRS_WGS84  = CRS.from_epsg(4326)


def km_to_deg_lat(km):
    return km / 111.32

def km_to_deg_lon(km, lat):
    import math
    return km / (111.32 * math.cos(math.radians(lat)))


def bbox_wgs84(lat, lon, largeur_km, hauteur_km):
    """Retourne (west, south, east, north) en WGS84."""
    dLat = km_to_deg_lat(hauteur_km / 2)
    dLon = km_to_deg_lon(largeur_km / 2, lat)
    return lon - dLon, lat - dLat, lon + dLon, lat + dLat


def trouver_dalles(data_dir: Path) -> list:
    """Trouve tous les fichiers MNT1m .asc dans le dossier data."""
    pattern = str(data_dir / "*/MNT1m/*.asc")
    dalles = sorted(glob.glob(pattern))
    return dalles


def fusionner(lat, lon, largeur_km, hauteur_km, sortie: Path):
    dalles = trouver_dalles(DATA_DIR)
    if not dalles:
        print(f"Erreur : aucun fichier .asc trouvé dans {DATA_DIR}/*/MNT1m/")
        return

    print(f"Dalles trouvées : {len(dalles)}")
    for d in dalles:
        print(f"  {Path(d).name}")

    # ── Ouverture de toutes les dalles avec CRS forcé ─────────────────────
    print("\nOuverture des dalles... ", end="", flush=True)
    datasets = []
    for d in dalles:
        src = rasterio.open(d)
        # Forcer Lambert-93 si absent
        if src.crs is None:
            from rasterio.io import MemoryFile
            profile = src.profile.copy()
            profile.update(crs=CRS_LAMB93)
            data = src.read()
            memfile = MemoryFile()
            with memfile.open(**profile) as mem:
                mem.write(data)
            datasets.append(memfile.open())
            src.close()
        else:
            datasets.append(src)
    print("OK")

    # ── Fusion ────────────────────────────────────────────────────────────
    print("Fusion des dalles... ", end="", flush=True)
    mosaic, transform = merge(datasets)
    print("OK")

    profile = datasets[0].profile.copy()
    profile.update({
        "crs":       CRS_LAMB93,
        "transform": transform,
        "width":     mosaic.shape[2],
        "height":    mosaic.shape[1],
        "count":     1,
        "driver":    "GTiff",
        "dtype":     mosaic.dtype,
    })

    for ds in datasets:
        ds.close()

    # ── Recadrage sur la zone d'intérêt ──────────────────────────────────
    print("Recadrage sur la zone... ", end="", flush=True)
    w_wgs, s_wgs, e_wgs, n_wgs = bbox_wgs84(lat, lon, largeur_km, hauteur_km)

    # Convertir le bbox WGS84 en Lambert-93
    from pyproj import Transformer
    t = Transformer.from_crs(CRS_WGS84, CRS_LAMB93, always_xy=True)
    x_min, y_min = t.transform(w_wgs, s_wgs)
    x_max, y_max = t.transform(e_wgs, n_wgs)

    # Marge de 100 m
    MARGE = 100
    x_min -= MARGE; y_min -= MARGE
    x_max += MARGE; y_max += MARGE

    geom = [box(x_min, y_min, x_max, y_max).__geo_interface__]

    # Écrire la mosaïque temporaire pour pouvoir la masquer
    tmp = Path("data/_tmp_mosaic.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(mosaic)

    with rasterio.open(tmp) as src:
        out_image, out_transform = mask(src, geom, crop=True)
        out_profile = src.profile.copy()
        out_profile.update({
            "transform": out_transform,
            "width":     out_image.shape[2],
            "height":    out_image.shape[1],
        })
    tmp.unlink()
    print("OK")

    # ── Écriture du GeoTIFF final ─────────────────────────────────────────
    sortie.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(sortie, "w", **out_profile) as dst:
        dst.write(out_image)

    print(f"\nFichier de sortie : {sortie}")
    print(f"  Dimensions : {out_image.shape[2]} × {out_image.shape[1]} px")
    print(f"  Superficie : {out_image.shape[2]/1000:.1f} × {out_image.shape[1]/1000:.1f} km")

    with rasterio.open(sortie) as src:
        valid = src.read(1, masked=True).compressed()
        print(f"  Altitude min : {valid.min():.2f} m")
        print(f"  Altitude max : {valid.max():.2f} m")
        b = src.bounds
        bw, bs, be, bn = transform_bounds(CRS_LAMB93, CRS_WGS84,
                                          b.left, b.bottom, b.right, b.top)
        print(f"  Emprise WGS84 : N={bn:.4f} S={bs:.4f} E={be:.4f} W={bw:.4f}")

    print("\nFusion terminée.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion des dalles Litto3D MNT1m")
    parser.add_argument("--lat",     type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",     type=float, default=DEFAULT_LON)
    parser.add_argument("--largeur", type=float, default=DEFAULT_LARGEUR)
    parser.add_argument("--hauteur", type=float, default=DEFAULT_HAUTEUR)
    parser.add_argument("--sortie",  type=str,   default=str(SORTIE))
    args = parser.parse_args()

    print(f"Zone : {args.lat}°N {abs(args.lon):.4f}°W  —  {args.largeur} × {args.hauteur} km\n")
    fusionner(args.lat, args.lon, args.largeur, args.hauteur, Path(args.sortie))
