# IMPORTS
import os
import io
import csv
import glob
import json
import shutil
from argparse import ArgumentParser

from PIL import Image


DEFAULT_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'experiment_config_template.json')


def _load_template(template_path):
    with open(template_path, 'r') as f:
        template = json.load(f)
    return {k: v for k, v in template.items() if not k.startswith('_')}


def _resolve_reference_dir(entry, key, references_root):
    """Resolve and validate the reference folder for this anatomy/projection.

    Returns (absolute reference dir, image file extension).
    """
    ref_subdir = entry.get('reference_dir', key)
    ref_dir = os.path.abspath(os.path.join(references_root, ref_subdir))
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(
            f"No reference set for '{key}' at {ref_dir}. Expected a folder with "
            f"*_image.* and *_landmarks.csv files."
        )
    # Recursive: supports a flat folder or per-case subfolders (e.g. <key>/patient1/*_image.*)
    ref_imgs = glob.glob(os.path.join(ref_dir, '**', '*_image.*'), recursive=True)
    if not ref_imgs:
        raise FileNotFoundError(f"Reference folder {ref_dir} contains no '*_image.*' files.")
    ext = os.path.splitext(ref_imgs[0])[1].lstrip('.')
    return ref_dir, ext


def _load_image(image):
    """Load the target image into a PIL Image.

    ``image`` may be raw bytes, a file-like object, or a path. This is the
    single seam where richer preprocessing belongs LATER (e.g. DICOM -> PNG
    conversion via pydicom, intensity windowing, resizing). For now only
    standard raster formats are handled.
    """
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    if hasattr(image, 'read'):
        return Image.open(image)
    # path
    if isinstance(image, str) and image.lower().endswith('.dcm'):
        # TODO: DICOM support — read pixel data with pydicom, apply windowing,
        # and (ideally) derive mpp from the PixelSpacing header here.
        raise NotImplementedError('DICOM input is not supported yet.')
    return Image.open(image)


def prepare_job(anatomy, projection, image, job_dir, references_root,
                mpp=None, num_references=None, max_matching_error=None,
                landmark_scaling=None, template_path=DEFAULT_TEMPLATE_PATH,
                target_id='target'):
    """Assemble everything a single matching+measurement job needs.

    Looks up the anatomy/projection template, resolves the reference set,
    preprocesses the target image, and writes a per-job config that is the
    single source of truth for both downstream steps.

    Any of ``mpp``, ``num_references``, ``max_matching_error`` and
    ``landmark_scaling`` override the template default for this job when given
    (``None`` = use the template value).

    Returns a dict of resolved paths/values for the orchestrator to use.
    """
    key = f'{anatomy}_{projection}'.lower()
    template = _load_template(template_path)
    if key not in template:
        raise ValueError(
            f"No template entry for '{key}'. Available: {sorted(template.keys())}"
        )
    entry = template[key]

    ref_dir, ext = _resolve_reference_dir(entry, key, references_root)

    # Per-job values: request override if given, else the template default
    def _pick(override, default):
        return default if override is None else override

    mpp_value = float(_pick(mpp, entry['mpp']))
    num_refs_value = int(_pick(num_references, entry.get('num_references', -1)))
    max_err_value = int(_pick(max_matching_error, entry.get('max_matching_error', 500)))
    scaling_value = list(_pick(landmark_scaling, entry.get('landmark_scaling', [1, 1])))

    # Job working directories
    input_dir = os.path.join(job_dir, 'input')
    output_dir = os.path.join(job_dir, 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess the target image. Re-encoding to the reference extension keeps
    # the file-type globbing in do_matching consistent for target + references.
    image_path = os.path.join(input_dir, f'{target_id}.{ext}')
    _load_image(image).save(image_path)

    # Per-job config (single source of truth). do_measurements reads mode + mpp
    # from the "job" tag; the orchestrator reads the matching knobs.
    job_entry = {
        'mode': entry['mode'],
        'mpp': mpp_value,
        'reference_dir': ref_dir,
        'num_references': num_refs_value,
        'max_matching_error': max_err_value,
        'landmark_scaling': scaling_value,
        'image_filetype': ext,
        'anatomy': anatomy,
        'projection': projection,
    }
    job_config_path = os.path.join(job_dir, 'job_config.json')
    with open(job_config_path, 'w') as f:
        json.dump({'job': job_entry}, f, indent=2)

    return {
        'job_dir': job_dir,
        'input_dir': input_dir,
        'output_dir': output_dir,
        'image_path': image_path,
        'image_filetype': ext,
        'target_id': target_id,
        'reference_dir': ref_dir,
        'job_config_path': job_config_path,
        'config_tag': 'job',
        'config': job_entry,
    }


if __name__ == '__main__':
    parser = ArgumentParser(description='Prepare a matching+measurement job (config + preprocessed image).')
    parser.add_argument('--anatomy', type=str, required=True, help='e.g. knee, feet, shoulder, hip')
    parser.add_argument('--projection', type=str, required=True, help='e.g. lateral, ap, axial')
    parser.add_argument('--image', type=str, required=True, help='Path to the target image')
    parser.add_argument('--job_dir', type=str, required=True, help='Output directory for this job')
    parser.add_argument('--references_root', type=str, required=True, help='Root holding per-anatomy/projection reference folders')
    parser.add_argument('--mpp', type=float, default=None, help='Millimeters per pixel (falls back to template default if unset)')
    parser.add_argument('--num_references', type=int, default=None, help='References to match against (-1 = all; falls back to template default)')
    parser.add_argument('--max_matching_error', type=int, default=None, help='Max allowed Procrustes error (falls back to template default)')
    parser.add_argument('--template_path', type=str, default=DEFAULT_TEMPLATE_PATH)
    args = parser.parse_args()

    result = prepare_job(
        anatomy=args.anatomy,
        projection=args.projection,
        image=args.image,
        job_dir=args.job_dir,
        references_root=args.references_root,
        mpp=args.mpp,
        num_references=args.num_references,
        max_matching_error=args.max_matching_error,
        template_path=args.template_path,
    )
    print(json.dumps(result, indent=2))
