#!/usr/bin/env python3
"""
Roma Medical - resident inference service.

A long-lived FastAPI service that loads the RoMa matching model once on
startup and keeps it resident on the GPU. Each request runs the full
preparation -> matching -> measurement pipeline for a single uploaded image
and returns both the consensus landmarks and the computed measurements.

The three stages run IN-PROCESS (not as subprocesses) so the resident model is
reused on every request:
    do_preparation.prepare_job()  -> per-job config + preprocessed image
    do_matching.main(.., MODEL)   -> consensus landmarks
    do_measurements.main(..)      -> measurements from the same job config

Endpoints:
    GET  /health    Liveness / readiness probe.
    POST /process   Run the pipeline on one uploaded image.
"""

# Ensure the project + ThirdParty are importable (mirrors the CLI scripts)
import sys
sys.path.append('./ThirdParty')

import os
import json
import glob
import logging
import tempfile
import threading
from argparse import Namespace

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

import do_preparation
import do_matching
import do_measurements
from utils.utils import read_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('roma_serve')

# ------------------------
# CONFIGURATION (via env)
# ------------------------
REFERENCES_ROOT = os.environ.get('REFERENCES_ROOT', '/refs')
TEMPLATE_PATH = os.environ.get('TEMPLATE_PATH', '/app/experiment_config_template.json')
COARSE_RES = tuple(int(x) for x in os.environ.get('COARSE_RES', '560,560').split(','))
UPSAMPLE_RES = tuple(int(x) for x in os.environ.get('UPSAMPLE_RES', '1120,1120').split(','))

# When API_KEY is set, /process requires a matching "X-API-Key" header.
# When unset, authentication is disabled (fine for local/trusted-only use).
API_KEY = os.environ.get('API_KEY')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The GPU model is not safe to call concurrently; serialize inference.
MODEL = None
MODEL_LOCK = threading.Lock()

app = FastAPI(title='Roma Medical', version='1.0')


@app.on_event('startup')
def load_model():
    """Load the RoMa model once when the service comes up."""
    global MODEL
    if not API_KEY:
        logger.warning('API_KEY not set -> /process is UNAUTHENTICATED (anyone who can reach '
                       'the port can use it). Set API_KEY for anything beyond local/trusted use.')
    logger.info('Loading RoMa model on %s ...', DEVICE)
    setup_hparams = Namespace(coarse_res=COARSE_RES, upsample_res=UPSAMPLE_RES)
    MODEL = do_matching.setup_model(setup_hparams, DEVICE)
    logger.info('Model loaded and resident.')


def _supported_keys():
    """anatomy_projection keys enabled in the template (those with references present)."""
    try:
        with open(TEMPLATE_PATH) as f:
            template = json.load(f)
    except (OSError, ValueError):
        return []
    keys = [k for k in template if not k.startswith('_')]
    available = []
    for k in sorted(keys):
        ref_dir = os.path.join(REFERENCES_ROOT, template[k].get('reference_dir', k))
        has_refs = bool(glob.glob(os.path.join(ref_dir, '**', '*_image.*'), recursive=True))
        available.append({'key': k, 'references_present': has_refs})
    return available


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
        'model_loaded': MODEL is not None,
        'references_root': REFERENCES_ROOT,
        'template_path': TEMPLATE_PATH,
        'auth_required': bool(API_KEY),
        'supported': _supported_keys(),
    }


def _run_pipeline(image_bytes, anatomy, projection, mpp, num_references, max_matching_error):
    """Run preparation + matching + measurement for one image."""
    with tempfile.TemporaryDirectory() as job_dir:

        # --- Preparation: template lookup, reference resolution, preprocessing,
        #     and the per-job config that is the single source of truth.
        #     Request overrides (or None -> template default) are layered in here ---
        try:
            job = do_preparation.prepare_job(
                anatomy=anatomy,
                projection=projection,
                image=image_bytes,
                job_dir=job_dir,
                references_root=REFERENCES_ROOT,
                mpp=mpp,
                num_references=num_references,
                max_matching_error=max_matching_error,
                template_path=TEMPLATE_PATH,
            )
        except (ValueError, NotImplementedError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

        cfg = job['config']
        target_id = job['target_id']

        # --- Matching (reuses the resident model; knobs come from the job config) ---
        match_hparams = Namespace(
            reference_path=cfg['reference_dir'],
            reference_rank_file=None,
            num_references=cfg['num_references'],
            data_path=job['input_dir'],
            save_path=job['output_dir'],
            image_filetype=cfg['image_filetype'],
            knn=15,
            max_matching_error=cfg['max_matching_error'],
            coarse_res=COARSE_RES,
            upsample_res=UPSAMPLE_RES,
            landmark_scaling=tuple(cfg['landmark_scaling']),
            temp_results=False,
            skip_selfmatching=False,
        )
        do_matching.main(match_hparams, roma_model=MODEL)

        # Consensus landmarks: <id>_matches_bulk.csv
        bulk_csv = os.path.join(job['output_dir'], f'{target_id}_matches_bulk.csv')
        if not os.path.exists(bulk_csv):
            raise HTTPException(
                status_code=422,
                detail='Matching did not produce a consensus result. This usually '
                       'indicates a laterality/anatomy mismatch with the reference set.',
            )
        landmarks = [[float(x), float(y)] for x, y in read_csv(bulk_csv)]

        # Per-reference Procrustes errors (matching confidence signal)
        procrustes = None
        procrustes_json = os.path.join(job['output_dir'], f'{target_id}_matches_bulk_procrustes.json')
        if os.path.exists(procrustes_json):
            with open(procrustes_json) as f:
                procrustes = json.load(f)

        # --- Measurements: reads mode + mpp straight from the job config ---
        measure_hparams = Namespace(
            data_path=job['output_dir'],
            save_path=job['output_dir'],
            config_path=job['job_config_path'],
            config_tag=job['config_tag'],
        )
        do_measurements.main(measure_hparams)

        measure_csv = os.path.join(job['output_dir'], f"measurements_{job['config_tag']}.csv")
        df = pd.read_csv(measure_csv, sep=';')
        row = df[df['ID'] == target_id].iloc[0].to_dict()
        row.pop('ID', None)
        measurements = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}

    confidence = None
    if procrustes:
        mean_err = float(np.mean(procrustes))
        max_err = cfg['max_matching_error']
        confidence = float(1 - np.clip(mean_err, 0, max_err) / max_err)

    return {
        'anatomy': anatomy,
        'projection': projection,
        'config_tag': cfg['mode'],
        'mpp': cfg['mpp'],
        'landmarks': landmarks,
        'measurements': measurements,
        'matching': {
            'num_references_used': len(procrustes) if procrustes else None,
            'mean_confidence': confidence,
            'procrustes_errors': procrustes,
        },
    }


@app.post('/process')
def process(
    image: UploadFile = File(..., description='Target radiograph (jpg/png)'),
    anatomy: str = Form(..., description='e.g. knee, feet, shoulder, hip'),
    projection: str = Form(..., description='e.g. lateral, ap, axial'),
    mpp: float = Form(None, description='Millimeters per pixel (default: template)'),
    num_references: int = Form(None, description='References to match against, -1 = all (default: template)'),
    max_matching_error: int = Form(None, description='Max allowed Procrustes error (default: template)'),
    x_api_key: str = Header(None, description='API key, required when the service has one configured'),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Missing or invalid API key (send it in the X-API-Key header).')

    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail='Empty image upload.')

    logger.info('Processing %s | %s/%s | mpp=%s num_references=%s max_matching_error=%s',
                image.filename, anatomy, projection, mpp, num_references, max_matching_error)
    with MODEL_LOCK:
        return _run_pipeline(image_bytes, anatomy, projection, mpp, num_references, max_matching_error)
