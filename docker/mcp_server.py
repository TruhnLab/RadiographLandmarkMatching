#!/usr/bin/env python3
"""
Remote MCP server (Streamable HTTP) that wraps the Roma Medical morphometry API.

It exposes MCP tools an agent can call to compute patellofemoral morphometry from
a knee radiograph. It does NO GPU work itself — it forwards to the morphometry
HTTP service internally and translates the outcome into a structured ``status``
so the agent can tell *why* a call failed (service down, anatomy not supported,
no match, ...) without any extra diagnostic plumbing.

Image transfer: the raw image (png/jpg/dcm) is sent once to ``POST /upload`` and
referenced by a short-lived token in the tool call — this keeps megabytes of
pixel data out of the agent's context. Small images may instead be passed inline
as base64. Runs behind the Caddy TLS proxy at paths ``/mcp``, ``/upload`` and
``/recompute`` (stateless recomputation from edited landmarks).
"""
import io
import os
import time
import base64
import secrets
import threading
from collections import OrderedDict

import httpx
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP
try:
    from mcp.server.fastmcp import Image          # image return type for MCP tools
except ImportError:                                # fallback for other SDK layouts
    from mcp.server.fastmcp.utilities.types import Image
from starlette.requests import Request
from starlette.responses import JSONResponse

from measurements.measurements import (get_function_dict, plot_image,
                                       measurement_kind, measurement_unit)

MORPH_URL = os.environ.get("MORPH_URL", "http://roma-medical:8000")
# key used to call the internal morphometry service (its own X-API-Key)
MORPH_API_KEY = os.environ.get("MORPH_API_KEY") or os.environ.get("API_KEY")
REQUEST_TIMEOUT = float(os.environ.get("MORPH_TIMEOUT", "300"))
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))  # 64 MB
UPLOAD_TTL = float(os.environ.get("UPLOAD_TTL", "600"))                      # 10 min
RESULT_TTL = float(os.environ.get("RESULT_TTL", "3600"))                     # 1 h
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "200"))                      # LRU cap

# token -> (image bytes, expiry timestamp). Single MCP-server instance -> in-memory is fine.
_STORE = {}
# result_token -> {image, landmarks, mode, mpp, measurements, exp}. Backs the PNG export
# and get_processed_image only; /recompute is stateless, so losing this costs a reader
# nothing. Bounded by MAX_RESULTS (LRU) so a long-lived server cannot grow unbounded.
_RESULTS = OrderedDict()
_RENDER_LOCK = threading.Lock()   # matplotlib global state is not thread-safe


def _purge():
    now = time.time()
    for k in [k for k, (_, exp) in _STORE.items() if exp < now]:
        _STORE.pop(k, None)
    for k in [k for k, v in _RESULTS.items() if v["exp"] < now]:
        _RESULTS.pop(k, None)


def _touch(token):
    """Fetch a cached result, refreshing its TTL and LRU position (None if absent)."""
    res = _RESULTS.get(token)
    if res is None:
        return None
    res["exp"] = time.time() + RESULT_TTL
    _RESULTS.move_to_end(token)
    return res


mcp = FastMCP("roma-morphometry", host="0.0.0.0", port=9000)


@mcp.custom_route("/upload", methods=["POST"])
async def upload(request: Request):
    """Accept a raw image (request body = file bytes) and return an image token.

    The token is single-use and expires after UPLOAD_TTL seconds. Pass it to
    ``get_morphometry(image_token=...)``.
    """
    data = await request.body()
    if not data:
        return JSONResponse({"error": "empty request body"}, status_code=400)
    if len(data) > MAX_UPLOAD:
        return JSONResponse({"error": f"file exceeds {MAX_UPLOAD} bytes"}, status_code=413)
    _purge()
    token = secrets.token_urlsafe(18)
    _STORE[token] = (data, time.time() + UPLOAD_TTL)
    return JSONResponse({"image_token": token, "bytes": len(data)})


def _supported():
    try:
        h = httpx.get(f"{MORPH_URL}/health", timeout=10).json()
        return [s["key"] for s in h.get("supported", []) if s.get("references_present")]
    except Exception:
        return []


def _resolve_token(image_token):
    """Look up (single-use) uploaded image bytes by their token."""
    _purge()
    entry = _STORE.pop(image_token, None)
    if entry is None:
        raise ValueError("unknown or expired image_token — upload the image again via POST /upload")
    return entry[0]


def _xy(point):
    """The vis_* helpers emit [row, col] = [y, x]; the wire uses [x, y].

    Returns None when the point is not finite. Degenerate landmarks (two that
    coincide, say, which is easy to produce by dragging one onto another) make
    the vis helpers emit NaN, and NaN is not valid JSON: serialising it fails
    the entire response instead of the one measurement that produced it, so the
    caller sees an empty body rather than a result.
    """
    x, y = float(point[1]), float(point[0])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None

    return [x, y]


def _segment(seg):
    """A drawable segment, or None when either endpoint is not finite."""
    start, end = _xy(seg[0]), _xy(seg[1])

    return None if start is None or end is None else [start, end]


def recompute_measurements(mode, mpp, landmarks, result_token=None):
    """Recompute measurements and construction geometry from landmarks. CPU only.

    Stateless: it needs nothing but ``mode``, ``mpp`` (both returned by get_morphometry)
    and the landmarks — no image, no GPU, no disk — so it is safe to call repeatedly in
    an interactive drag loop and survives a server restart.

    ``landmarks`` are [x, y] in the frame of the *processed* image. Values reproduce
    do_measurements exactly (measure on [y, x] * mpp); geometry comes straight from the
    vis_* helpers and is converted back to [x, y], so there is one convention outward.

    If ``result_token`` is given and still cached, the stored landmarks are updated so a
    later visualize_measurement renders the corrected construction. Best effort only: an
    expired token is ignored and the recomputation still succeeds.
    """
    try:
        fdict, kpt_names = get_function_dict(mode=mode)
    except Exception as e:
        return {"status": "bad_input", "detail": f"unknown mode '{mode}': {e}"}
    if not fdict:
        return {"status": "bad_input", "detail": f"unknown mode '{mode}'"}

    try:
        mpp = float(mpp)
    except (TypeError, ValueError):
        return {"status": "bad_input", "detail": "mpp must be a number"}
    if not np.isfinite(mpp) or mpp <= 0:
        return {"status": "bad_input", "detail": "mpp must be a positive, finite number"}

    try:
        lm = np.asarray(landmarks, dtype=float)
    except (TypeError, ValueError):
        return {"status": "bad_input", "detail": "landmarks must be a list of [x, y] numbers"}
    if lm.ndim != 2 or lm.shape[0] == 0 or lm.shape[1] != 2:
        return {"status": "bad_input",
                "detail": "landmarks must be a non-empty list of [x, y] pairs"}
    if not np.isfinite(lm).all():
        return {"status": "bad_input", "detail": "landmarks contain NaN or infinite values"}
    expected = len(kpt_names) if kpt_names else 0
    if expected and lm.shape[0] != expected:
        return {"status": "bad_input",
                "detail": f"mode '{mode}' expects {expected} landmarks, got {lm.shape[0]}"}

    lm_yx = lm[:, ::-1]          # [x, y] -> [y, x] for the measurement / vis helpers
    scaled = lm_yx * mpp         # exactly what do_measurements measures on

    measurements, geometry = {}, {}
    for name, (measure_fn, vis_fn, kpts_fn) in fdict.items():
        try:
            value = float(measure_fn(scaled))
            if not np.isfinite(value):
                value = None
        except Exception:
            value = None
        measurements[name] = value

        entry = {"value": value, "kind": measurement_kind(name), "unit": measurement_unit(name),
                 "landmark_indices": [], "points": [], "solid_lines": [], "dotted_lines": []}
        try:
            entry["landmark_indices"] = [int(i) for i in kpts_fn()]
        except Exception:
            pass
        try:
            g = vis_fn(lm_yx)    # [points, solid_lines, dotted_lines, *colours] in [y, x]
            # Non-finite coordinates are dropped rather than emitted: they cannot be
            # drawn, and one of them would otherwise take the whole response with it.
            entry["points"] = [p for p in (_xy(p) for p in g[0]) if p is not None]
            entry["solid_lines"] = [s for s in (_segment(seg) for seg in g[1]) if s is not None]
            entry["dotted_lines"] = [s for s in (_segment(seg) for seg in g[2]) if s is not None]
        except Exception:
            pass
        geometry[name] = entry

    if result_token:             # best effort: keep the PNG export in sync with the edits
        res = _touch(result_token)
        if res is not None:
            res["landmarks"] = [[float(x), float(y)] for x, y in lm]
            res["measurements"] = measurements

    return {"status": "ok", "measurements": measurements, "geometry": geometry}


@mcp.custom_route("/recompute", methods=["POST"])
async def recompute(request: Request):
    """Recompute measurements + geometry from edited landmarks (stateless, CPU only).

    Body: {"mode": "knee_lateral", "mpp": 0.148, "landmarks": [[x, y], ...],
           "result_token": "<optional>"}

    Exposed as a plain HTTP route (not only an MCP tool) because it sits in an
    interactive drag loop, where a Streamable-HTTP handshake per call would be too slow.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "bad_input", "detail": "body must be JSON"},
                            status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"status": "bad_input", "detail": "body must be a JSON object"},
                            status_code=400)
    out = recompute_measurements(body.get("mode"), body.get("mpp"),
                                 body.get("landmarks"), body.get("result_token"))
    return JSONResponse(out, status_code=200 if out.get("status") == "ok" else 400)


@mcp.tool()
def get_morphometry(anatomy: str, projection: str, image_token: str,
                    mpp: float | None = None, num_references: int | None = None) -> dict:
    """Compute morphometry measurements from a single knee radiograph.

    First upload the raw image (png/jpg/DICOM) to `POST /upload`, which returns an
    `image_token`; pass that token here. `anatomy`/`projection` pick the reference
    set (currently supported: knee/lateral, knee/axial — call `morphometry_health`
    to check). `mpp` (mm per pixel) and `num_references` are optional overrides.

    Always returns a dict with a `status` field:
      - "ok"          -> also has `measurements`, `landmarks`, `matching`, `mode` and
                         `mpp` (pass these to `/recompute` / `update_landmarks` to
                         recompute from edited landmarks), `laterality_corrected` (True
                         when matching had to mirror a working copy to align with the
                         reference set; the landmarks are mapped back either way),
                         and a `result_token` for `visualize_measurement` /
                         `get_processed_image`
      - "unsupported" -> anatomy/projection not enabled; see `supported`
      - "no_match"    -> matching failed (commonly wrong laterality/anatomy)
      - "unavailable" -> morphometry service unreachable or not ready yet
      - "bad_input"   -> image_token invalid/expired or arguments invalid (see `detail`)
      - "error"       -> unexpected failure (see `detail`)
    """
    try:
        img = _resolve_token(image_token)
    except Exception as e:
        return {"status": "bad_input", "detail": str(e)}

    data = {"anatomy": anatomy, "projection": projection, "return_image": "true"}
    if mpp is not None:
        data["mpp"] = str(mpp)
    if num_references is not None:
        data["num_references"] = str(num_references)
    headers = {"X-API-Key": MORPH_API_KEY} if MORPH_API_KEY else {}

    try:
        r = httpx.post(f"{MORPH_URL}/process", data=data,
                       files={"image": ("image", img)}, headers=headers,
                       timeout=REQUEST_TIMEOUT)
    except httpx.RequestError as e:
        return {"status": "unavailable",
                "detail": f"morphometry service not reachable: {e}"}

    if r.status_code == 200:
        d = r.json()
        measurements = d.get("measurements") or {}
        mode, mpp = d.get("config_tag"), d.get("mpp")
        # Cache the (processed) image + landmarks so an overlay can be rendered later.
        result_token = None
        img_b64 = d.pop("image_b64", None)
        if img_b64:
            _purge()
            result_token = secrets.token_urlsafe(18)
            _RESULTS[result_token] = {
                "image": base64.b64decode(img_b64),
                "landmarks": d.get("landmarks"),
                "mode": mode,
                "mpp": mpp,
                "measurements": measurements,
                "exp": time.time() + RESULT_TTL,
            }
            while len(_RESULTS) > MAX_RESULTS:      # LRU: evict the least recently used
                _RESULTS.popitem(last=False)
        return {"status": "ok",
                "anatomy": anatomy, "projection": projection,
                "mode": mode, "mpp": mpp,
                "measurements": measurements,
                "landmarks": d.get("landmarks"),
                "matching": d.get("matching"),
                "laterality_corrected": d.get("laterality_corrected"),
                "result_token": result_token,
                "visualizable_measurements": list(measurements.keys())}

    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    if r.status_code == 400:
        return {"status": "unsupported", "detail": detail, "supported": _supported()}
    if r.status_code == 422:
        return {"status": "no_match", "detail": detail}
    if r.status_code == 401:
        return {"status": "error", "detail": "morphometry API key rejected (server config issue)"}
    return {"status": "error", "http_status": r.status_code, "detail": detail}


@mcp.tool()
def update_landmarks(mode: str, mpp: float, landmarks: list,
                     result_token: str | None = None) -> dict:
    """Recompute measurements and construction geometry from (corrected) landmarks.

    The stateless, CPU-only counterpart to get_morphometry's measurement step: no GPU and
    no re-matching. Give it the `mode` and `mpp` from a get_morphometry result plus the
    landmarks as [x, y] pairs (e.g. after a reader corrected some), and it returns the
    updated `measurements` and, per measurement, the `geometry` needed to draw the
    construction: `points`, `solid_lines`, `dotted_lines` (all [x, y]), plus
    `landmark_indices`, `kind` and `unit`.

    Pass the optional `result_token` to also update that result's stored landmarks, so a
    later visualize_measurement renders the corrected construction (ignored if expired).
    Returns {status:"ok", measurements, geometry} or {status:"bad_input", detail}.
    """
    return recompute_measurements(mode, mpp, landmarks, result_token)


@mcp.tool()
def morphometry_health() -> dict:
    """Discover the full capability set, without running the GPU.

    Returns {status: "ok"|"loading"|"unavailable", supported: [...], detail?}.
    Each `supported` entry is {anatomy, projection, mode, measurements: [...],
    landmark_names: [...]} for a combination whose reference set is present. Everything
    here is static per mode, so fetch it once and cache it: an agent (or UI) learns which
    anatomy/projection pairs work, exactly which measurements each yields (the names
    `visualize_measurement` accepts), and a label for every landmark index.
    """
    try:
        h = httpx.get(f"{MORPH_URL}/health", timeout=10).json()
    except Exception as e:
        return {"status": "unavailable", "detail": str(e)}
    supported = []
    for s in h.get("supported", []):
        if not s.get("references_present"):
            continue
        key = s.get("key", "")
        mode = s.get("mode", key)
        anatomy, _, projection = key.partition("_")
        measurements, landmark_names = [], []
        try:
            fdict, kpt_names = get_function_dict(mode=mode)
            measurements = list(fdict.keys()) if fdict else []
            landmark_names = [n[1] if isinstance(n, (list, tuple)) and len(n) > 1 else str(n)
                              for n in (kpt_names or [])]
        except Exception:
            pass
        supported.append({"anatomy": anatomy, "projection": projection, "mode": mode,
                          "measurements": measurements, "landmark_names": landmark_names})
    return {"status": "ok" if h.get("model_loaded") else "loading",
            "supported": supported}


@mcp.tool()
def visualize_measurement(result_token: str, measurement: str):
    """Return an image of one measurement drawn on the radiograph.

    Use the `result_token` from a successful `get_morphometry` call together with
    one of its measurement names (see that call's `visualizable_measurements`).
    Returns the processed image with that measurement's construction (points and
    lines) overlaid; the same token can be reused for different measurements until
    it expires. On a problem it returns a dict with a `status` ("bad_input" for an
    unknown/expired token or unknown measurement).
    """
    _purge()
    res = _touch(result_token)
    if res is None:
        return {"status": "bad_input",
                "detail": "unknown or expired result_token — run get_morphometry again"}
    try:
        fdict, _ = get_function_dict(mode=res["mode"])
    except Exception as e:
        return {"status": "error", "detail": f"no visualizations for mode '{res['mode']}': {e}"}
    if measurement not in fdict:
        return {"status": "bad_input", "detail": f"unknown measurement '{measurement}'",
                "available": list(fdict.keys())}

    vis_func = fdict[measurement][1]
    # landmarks come back as [x, y]; the vis/plot helpers use [row, col] = [y, x]
    lm = np.array([[float(y), float(x)] for x, y in res["landmarks"]])
    image = np.array(PILImage.open(io.BytesIO(res["image"])).convert("L"))
    value = res["measurements"].get(measurement)
    value = float(value) if isinstance(value, (int, float)) else None

    with _RENDER_LOCK:
        try:
            plt.close("all")
            plot_image(image, vis_func(lm), name=measurement, value=value, linewidth=3)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        finally:
            plt.close("all")
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def get_processed_image(result_token: str):
    """Return the exact radiograph that landmarks and measurements were computed on.

    This is the prepared image (e.g. re-encoded to the reference set's format), the frame
    the `landmarks` from the same get_morphometry call live in. Matching never modifies
    it: when it corrects laterality it mirrors a working copy and maps the landmarks back
    (`get_morphometry` reports this as `laterality_corrected`). Use it for any downstream
    work that must line up with the returned landmarks/measurements. Pass the
    `result_token` from a successful get_morphometry; it stays valid until the result
    expires. On a bad or expired token it returns a dict {status:"bad_input"}.
    """
    _purge()
    res = _touch(result_token)
    if res is None:
        return {"status": "bad_input",
                "detail": "unknown or expired result_token — run get_morphometry again"}
    data = res["image"]
    fmt = "png"
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            fmt = (im.format or "PNG").lower()
        fmt = {"jpg": "jpeg", "jpe": "jpeg"}.get(fmt, fmt)
    except Exception:
        pass
    return Image(data=data, format=fmt)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
