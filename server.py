#!/usr/bin/env python3
"""
server.py — Serveur web local pour le pipeline carte3d
Usage : python server.py
        Puis ouvrir http://localhost:5000
"""
import json
import threading
import traceback
import uuid
from pathlib import Path

import matplotlib
matplotlib.use('Agg')             # backend non-interactif avant tout import matplotlib

from flask import Flask, jsonify, redirect, render_template, request, send_file
import numpy as np
import rasterio

import carte3d

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')

SESSION_FILE = Path('session.json')
STATIC_DIR   = Path('static')
STATIC_DIR.mkdir(exist_ok=True)

JOBS: dict = {}   # job_id → {done, total, log, finished, error, result}

# ── Session ────────────────────────────────────────────────────────────────────
DEFAULT_SESSION = dict(
    mnt           = carte3d.MNT_DEFAUT,
    lat           = 48.841,
    lon           = -3.302,
    largeur       = 4.3,
    hauteur       = 3.8,
    carte_w       = 430,
    carte_h       = 380,
    echelle       = 10000,
    equidistance  = 5,
    epaisseur     = 3.0,
    seuil_surface = 2000.0,
    fraise_mm     = 3.0,
    angle_fraise_v= 45.0,
    dossier_sortie= carte3d.SORTIE_DEFAUT,
)

def load_session() -> dict:
    s = DEFAULT_SESSION.copy()
    if SESSION_FILE.exists():
        s.update(json.loads(SESSION_FILE.read_text()))
    return s

def save_session(data: dict) -> dict:
    s = load_session()
    s.update(data)
    SESSION_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False))
    return s

def session_to_args(s: dict):
    """Convertit la session en args pour carte3d (filtre les clés inconnues)."""
    keys = ['mnt','lat','lon','largeur','hauteur','echelle','equidistance',
            'epaisseur','seuil_surface','angle_fraise_v','fraise_mm',
            'simplification','lissage','dossier_sortie']
    return carte3d.make_args(**{k: s[k] for k in keys if k in s})

# ── Cache MNT ──────────────────────────────────────────────────────────────────
_mnt_cache: dict = {}
_mnt_lock  = threading.Lock()

def get_mnt(mnt_path: str) -> dict:
    with _mnt_lock:
        if _mnt_cache.get('path') == mnt_path:
            return _mnt_cache
        p = Path(mnt_path)
        if not p.exists():
            raise FileNotFoundError(f"MNT introuvable : {mnt_path}")
        with rasterio.open(p) as src:
            data_raw         = src.read(1).astype(float)
            nodata           = src.nodata
            raster_transform = src.transform
            bounds           = src.bounds
        data = (np.ma.masked_where(data_raw == nodata, data_raw)
                if nodata is not None else np.ma.array(data_raw))
        valid = data.compressed()
        _mnt_cache.update(path=mnt_path, data=data,
                          raster_transform=raster_transform, bounds=bounds,
                          z_min=float(valid.min()), z_max=float(valid.max()))
        return _mnt_cache

def load_mnt_and_couches(session: dict):
    mnt    = get_mnt(session['mnt'])
    args   = session_to_args(session)
    couches = carte3d.definir_couches(mnt['z_min'], mnt['z_max'], args.equidistance)
    return mnt, args, couches

# ── Job helpers ────────────────────────────────────────────────────────────────
def new_job() -> str:
    jid = str(uuid.uuid4())[:8]
    JOBS[jid] = dict(done=0, total=0, log=[], finished=False, error=None, result=None)
    return jid

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('zone.html', session=load_session())

@app.route('/params')
def params():
    return render_template('params.html', session=load_session())

@app.route('/generate')
def generate_page():
    return render_template('generate.html', session=load_session())

# ── API : session ──────────────────────────────────────────────────────────────
@app.route('/api/zone', methods=['POST'])
def api_zone():
    try:
        return jsonify(ok=True, session=save_session(request.json))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/params', methods=['POST'])
def api_params():
    try:
        return jsonify(ok=True, session=save_session(request.json))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/session')
def api_session():
    return jsonify(load_session())

# ── API : analyse ──────────────────────────────────────────────────────────────
@app.route('/api/analyse', methods=['POST'])
def api_analyse():
    session = save_session(request.json or {})
    jid = new_job()

    def run():
        try:
            mnt, args, couches = load_mnt_and_couches(session)
            n_layers = sum(1 for c in couches if c.materiau != 'plexi')
            JOBS[jid]['total'] = n_layers
            result = carte3d.run_analyse_data(
                args, mnt['data'], mnt['raster_transform'], mnt['bounds'], couches,
                progress_cb=lambda n: JOBS[jid].update(done=n)
            )
            JOBS[jid]['result'] = result
        except Exception:
            JOBS[jid]['error'] = traceback.format_exc()
        finally:
            JOBS[jid]['finished'] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify(job_id=jid)

# ── API : preview ──────────────────────────────────────────────────────────────
@app.route('/api/preview', methods=['POST'])
def api_preview():
    session = save_session(request.json or {})
    jid = new_job()
    JOBS[jid]['total'] = 1

    def run():
        try:
            mnt, args, couches = load_mnt_and_couches(session)
            args.png            = True
            args.dossier_sortie = str(STATIC_DIR)
            carte3d.run_preview(args, mnt['data'], mnt['raster_transform'],
                                mnt['bounds'], couches)
            JOBS[jid]['result'] = dict(url='/static/preview.png')
        except Exception:
            JOBS[jid]['error'] = traceback.format_exc()
        finally:
            JOBS[jid]['done']     = 1
            JOBS[jid]['finished'] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify(job_id=jid)

# ── API : generate ─────────────────────────────────────────────────────────────
@app.route('/api/generate', methods=['POST'])
def api_generate():
    session = save_session(request.json or {})
    jid = new_job()

    def run():
        try:
            mnt, args, couches = load_mnt_and_couches(session)
            JOBS[jid]['total'] = len(couches)

            def progress_cb(i, total, nom, n_pieces):
                JOBS[jid]['done'] = i
                JOBS[jid]['log'].append(dict(num=i, nom=nom, n=n_pieces))

            carte3d.run_generate(args, mnt['data'], mnt['raster_transform'],
                                 mnt['bounds'], couches, progress_cb=progress_cb)

            sortie = Path(args.dossier_sortie)
            JOBS[jid]['result'] = dict(
                n_files       = len(list(sortie.glob('*.svg'))),
                dossier       = str(sortie.resolve()),
                total_pieces  = sum(e['n'] for e in JOBS[jid]['log']),
            )
        except Exception:
            JOBS[jid]['error'] = traceback.format_exc()
        finally:
            JOBS[jid]['finished'] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify(job_id=jid)

# ── API : status ───────────────────────────────────────────────────────────────
@app.route('/api/status/<jid>')
def api_status(jid):
    job = JOBS.get(jid)
    return (jsonify(job) if job else (jsonify(error='job introuvable'), 404))

# ── Lancement ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\nCarte 3D — Serveur local")
    print("─" * 30)
    print("  http://localhost:5000")
    print("  Ctrl+C pour arrêter\n")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
