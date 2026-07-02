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
import os
import time
import secrets

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

MORPH_URL = os.environ.get("MORPH_URL", "http://roma-medical:8000")
# key used to call the internal morphometry service (its own X-API-Key)
MORPH_API_KEY = os.environ.get("MORPH_API_KEY") or os.environ.get("API_KEY")
REQUEST_TIMEOUT = float(os.environ.get("MORPH_TIMEOUT", "300"))
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))  # 64 MB
UPLOAD_TTL = float(os.environ.get("UPLOAD_TTL", "600"))                      # 10 min

# token -> (image bytes, expiry timestamp). Single MCP-server instance -> in-memory is fine.
_STORE = {}


def _purge():
    now = time.time()
    for k in [k for k, (_, exp) in _STORE.items() if exp < now]:
        _STORE.pop(k, None)


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
      - "ok"          -> also has `measurements`, `landmarks`, `matching`
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

    data = {"anatomy": anatomy, "projection": projection}
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
        return {"status": "ok",
                "anatomy": anatomy, "projection": projection,
                "measurements": d.get("measurements"),
                "landmarks": d.get("landmarks"),
                "matching": d.get("matching")}

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
    """Check the morphometry service before/without running a measurement.

    Returns {status: "ok"|"loading"|"unavailable", supported: [...], detail?}
    where `supported` lists the anatomy_projection combinations that are ready.
    """
    try:
        h = httpx.get(f"{MORPH_URL}/health", timeout=10).json()
    except Exception as e:
        return {"status": "unavailable", "detail": str(e)}
    return {"status": "ok" if h.get("model_loaded") else "loading",
            "supported": [s["key"] for s in h.get("supported", []) if s.get("references_present")]}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
