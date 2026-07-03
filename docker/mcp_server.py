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
as base64. Runs behind the Caddy TLS proxy at paths ``/mcp`` and ``/upload``.
"""
import io
import os
import time
import base64
import secrets
import threading

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

from measurements.measurements import get_function_dict, plot_image

MORPH_URL = os.environ.get("MORPH_URL", "http://roma-medical:8000")
# key used to call the internal morphometry service (its own X-API-Key)
MORPH_API_KEY = os.environ.get("MORPH_API_KEY") or os.environ.get("API_KEY")
REQUEST_TIMEOUT = float(os.environ.get("MORPH_TIMEOUT", "300"))
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))  # 64 MB
UPLOAD_TTL = float(os.environ.get("UPLOAD_TTL", "600"))                      # 10 min
RESULT_TTL = float(os.environ.get("RESULT_TTL", "900"))                      # 15 min

# token -> (image bytes, expiry timestamp). Single MCP-server instance -> in-memory is fine.
_STORE = {}
# result_token -> {image, landmarks, mode, measurements, exp} for measurement overlays.
_RESULTS = {}
_RENDER_LOCK = threading.Lock()   # matplotlib global state is not thread-safe


def _purge():
    now = time.time()
    for k in [k for k, (_, exp) in _STORE.items() if exp < now]:
        _STORE.pop(k, None)
    for k in [k for k, v in _RESULTS.items() if v["exp"] < now]:
        _RESULTS.pop(k, None)


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


@mcp.tool()
def get_morphometry(anatomy: str, projection: str, image_token: str,
                    mpp: float | None = None, num_references: int | None = None) -> dict:
    """Compute morphometry measurements from a single knee radiograph.

    First upload the raw image (png/jpg/DICOM) to `POST /upload`, which returns an
    `image_token`; pass that token here. `anatomy`/`projection` pick the reference
    set (currently supported: knee/lateral, knee/axial — call `morphometry_health`
    to check). `mpp` (mm per pixel) and `num_references` are optional overrides.

    Always returns a dict with a `status` field:
      - "ok"          -> also has `measurements`, `landmarks`, `matching`, and a
                         `result_token` you can pass to `visualize_measurement`
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
        # Cache the (processed) image + landmarks so an overlay can be rendered later.
        result_token = None
        img_b64 = d.pop("image_b64", None)
        if img_b64:
            _purge()
            result_token = secrets.token_urlsafe(18)
            _RESULTS[result_token] = {
                "image": base64.b64decode(img_b64),
                "landmarks": d.get("landmarks"),
                "mode": d.get("config_tag"),
                "measurements": measurements,
                "exp": time.time() + RESULT_TTL,
            }
        return {"status": "ok",
                "anatomy": anatomy, "projection": projection,
                "measurements": measurements,
                "landmarks": d.get("landmarks"),
                "matching": d.get("matching"),
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
def morphometry_health() -> dict:
    """Discover the full capability set, without running the GPU.

    Returns {status: "ok"|"loading"|"unavailable", supported: [...], detail?}.
    Each `supported` entry is {anatomy, projection, measurements: [...]} for a
    combination whose reference set is present, so an agent can learn up front which
    anatomy/projection pairs work and exactly which measurements each one yields
    (the same names accepted by `visualize_measurement`).
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
        anatomy, _, projection = key.partition("_")
        try:
            fdict, _ = get_function_dict(mode=s.get("mode", key))
            measurements = list(fdict.keys()) if fdict else []
        except Exception:
            measurements = []
        supported.append({"anatomy": anatomy, "projection": projection,
                          "measurements": measurements})
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
    res = _RESULTS.get(result_token)
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


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
