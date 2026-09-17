"""RoMa matching of one target against many references, without repeated work.

``compute_matches`` (utils.utils) handles one (reference, target) pair from
file paths: both images are decoded and resized, and the full encoder runs,
for every pair. With N references per target that repeats the target work N
times and the reference work N times per target.

``MatchingEngine`` produces bit-identical per-pair results with the work split
by what it depends on:

* references: decoded, resized and DINOv2-encoded once, then cached (keyed by
  path, modification time and size, so an edited reference is re-encoded).
  Resized images stay in CPU memory, DINOv2 features on the GPU (~3 MB each);
  the cheap CNN features are recomputed when a reference is matched.
* target:     decoded, resized and encoded once per image
* pairs:      decoded by RoMa one pair at a time; sampling and keypoint transfer
              use the same per-pair seed as compute_matches

Bit-identity constraints (checked by regression/check_components.py): cuDNN
picks convolution algorithms by input shape, so CNN feature maps match the
original pipeline, which encodes (reference, target) as a batch of 2, only
when they are computed in a batch of 2 as well. The per-image result does not
depend on the batch partner, so references are encoded in pairs and a single
image is paired with itself. The decoder is likewise run on exactly one pair.
DINOv2 features do not depend on the batch size.

The engine is kept per model (``get_engine``), so a resident service reuses the
reference cache across requests.
"""

import os
import numpy as np
import torch
from PIL import Image

from utils.utils import read_csv, pair_seed

# References handed to MatchingEngine.match per call (CNN encoding batch, see above)
REFERENCES_PER_CALL = 2


def register_keypoints(keypoints_source, kpts1, kpts2, knn=15):
    """Transfer landmarks by the mean displacement of their ``knn`` nearest sampled matches.

    Vectorised form of the per-landmark loop in compute_matches, with the same
    dtypes (float64 landmarks, float32 matches) and operation order.
    """
    dist = torch.norm(kpts1[None] - keypoints_source[:, None], dim=2, p=None)
    nearest = dist.topk(knn, dim=1, largest=False).indices
    translation = torch.mean(kpts1[nearest] - kpts2[nearest], dim=1)
    return keypoints_source - translation


def load_keypoints(kpt_path, landmark_scaling=(1, 1)):
    keypoints = read_csv(kpt_path)
    keypoints = np.array([[float(x)*landmark_scaling[1],
                           float(y)*landmark_scaling[0]] for x, y in keypoints])
    return torch.tensor(keypoints)


def _in_pairs(images, encode):
    """Run ``encode`` on batches of exactly 2 images (the last one padded with itself)
    and return one feature dict per image."""
    out = []
    for i in range(0, len(images), 2):
        chunk = images[i:i + 2]
        batch = torch.cat(chunk + chunk[:1] * (2 - len(chunk)))
        feats = encode(batch)
        out += [{s: f[j:j + 1] for s, f in feats.items()} for j in range(len(chunk))]
    return out


class MatchingEngine:

    def __init__(self, model, device):
        self.model = model
        self.device = torch.device(device)
        self._refs = {}  # one reference set per use case (~100 images), so no eviction

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def _load(self, path):
        """Decode + resize; uint8 tensors on the CPU (pinned when a GPU is used)."""
        with Image.open(path) as im:
            cols, rows = im.size
            coarse, fine = self.model.resize_image(im)
        if self.device.type == 'cuda':
            coarse, fine = coarse.pin_memory(), fine.pin_memory()
        return rows, cols, coarse, fine

    def _normalized(self, image_uint8):
        return self.model.normalize_images(image_uint8[None].to(self.device, non_blocking=True))

    @staticmethod
    def _cache_key(path):
        st = os.stat(path)
        return os.path.realpath(path), st.st_mtime_ns, st.st_size

    @torch.inference_mode()
    def cache_references(self, ref_img_paths):
        """Make sure all references are cached; returns their cache entries in order."""
        keys = [self._cache_key(p) for p in ref_img_paths]
        missing = [(k, p) for k, p in zip(keys, ref_img_paths) if k not in self._refs]
        for i in range(0, len(missing), 2):
            chunk = missing[i:i + 2]
            loaded = [self._load(p) for _, p in chunk]
            dino = _in_pairs([self._normalized(l[2]) for l in loaded],
                             lambda x: {16: self.model.encode_dino(x)})
            for (key, path), (rows, cols, coarse, fine), d in zip(chunk, loaded, dino):
                self._refs[key] = {'path': path, 'rows': rows, 'cols': cols,
                                   'coarse': coarse, 'fine': fine, 'dino': d[16].clone()}
        return [self._refs[key] for key in keys]

    @torch.inference_mode()
    def prepare_target(self, img_path):
        rows, cols, coarse, fine = self._load(img_path)
        model = self.model
        return {
            'path': img_path, 'rows': rows, 'cols': cols,
            'coarse': _in_pairs([self._normalized(coarse)], lambda x: model.encoder(x, upsample=False))[0],
            'fine': _in_pairs([self._normalized(fine)], model.encode_cnn)[0],
        }

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def match(self, target, ref_img_paths, ref_kpt_paths, landmark_scaling=(1, 1), knn=15):
        """Match ``target`` (from prepare_target) against references, in order.

        Pass REFERENCES_PER_CALL references per call to share their CNN encoding.
        Returns one (keypoints_source, registered_kpts) tuple per reference, equal
        to what compute_matches returns (float64 CPU tensors).
        """
        model = self.model
        refs = self.cache_references(ref_img_paths)
        coarse = _in_pairs([self._normalized(r['coarse']) for r in refs], model.encode_cnn)
        fine = _in_pairs([self._normalized(r['fine']) for r in refs], model.encode_cnn)

        results = []
        for ref, ref_coarse, ref_fine, ref_img_path, ref_kpt_path in zip(refs, coarse, fine, ref_img_paths, ref_kpt_paths):
            keypoints_source = load_keypoints(ref_kpt_path, landmark_scaling)
            warp, certainty = model.match_from_features(
                {'coarse': {**ref_coarse, 16: ref['dino']}, 'fine': ref_fine},
                {'coarse': target['coarse'], 'fine': target['fine']})

            # Same per-pair seed as compute_matches; the network itself draws no random numbers
            seed = pair_seed(ref_img_path, target['path'])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            matches, _ = model.sample(warp[0], certainty[0])
            kpts1, kpts2 = model.to_pixel_coordinates(matches, ref['rows'], ref['cols'],
                                                      target['rows'], target['cols'])
            registered_kpts = register_keypoints(keypoints_source, kpts1.cpu(), kpts2.cpu(), knn=knn)
            results.append((keypoints_source, registered_kpts))
        return results


def get_engine(model, device):
    """The engine (and so the reference cache) belonging to ``model``."""
    engine = getattr(model, '_matching_engine', None)
    if engine is None:
        engine = MatchingEngine(model, device)
        object.__setattr__(model, '_matching_engine', engine)  # plain attribute, not a registered submodule
    return engine
