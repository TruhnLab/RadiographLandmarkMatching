# Roma Medical: Dockerized inference service

Run the morphometry pipeline yourself as a self-hosted service on your own GPU
machine. The RoMa model loads once and stays in GPU memory, then clients send a
radiograph with `anatomy` and `projection` (and optional `mpp`) and get back the
consensus landmarks and clinical measurements. It is reachable two ways: a REST API,
and a remote MCP server for agents. Both sit behind an HTTPS proxy with API-key auth.

Each request runs three in-process stages, reusing the resident model: preparation
(`do_preparation.py`), matching (`do_matching.py`), measurement (`do_measurements.py`).

## Requirements

- An NVIDIA GPU (about 24 GB; the model uses ~23 GB at the default resolution), a
  recent driver, Docker, and the NVIDIA Container Toolkit. CPU works for smoke tests
  but is slow.
- The two weight files `roma_outdoor.pth` and `dinov2_vitl14_pretrain.pth` (mounted,
  not baked into the image).
- Reference sets (images + landmark CSVs) for each anatomy/projection you enable.

The `Dockerfile` uses CUDA `cu121`; change the base image and torch index if your
driver needs a different version.

## Data you provide (mounted at runtime)

Keep everything in one folder, called your data root below:

```
<data-root>/
├── model_weights/           # roma_outdoor.pth, dinov2_vitl14_pretrain.pth
├── refs/                    # one subfolder per <anatomy>_<projection>
└── experiment_config_template.json
```

| Host | Container path |
|------|----------------|
| `model_weights/` | `/app/ThirdParty/model_weights` |
| `refs/` | `/refs` |
| `experiment_config_template.json` | `/app/experiment_config_template.json` |

### Reference layout

One folder per `anatomy_projection`. Inside, the `*_image.*` and `*_landmarks.csv`
pairs may be flat or in per-case subfolders (both are found recursively):

```
refs/
├── knee_lateral/
│   ├── case1/{case1_image.jpg, case1_landmarks.csv}
│   └── case2/ ...
└── knee_axial/
    ├── 0001_image.jpg
    └── 0001_landmarks.csv
```

The prefix before `_image` and `_landmarks` must match. All images in a set share
one extension (jpg or png). References are single-orientation. The folder name must
match a tag in `experiment_config_template.json`.

## Templates

`experiment_config_template.json` holds one entry per supported `anatomy_projection`
with its static settings (`mode`, default `mpp`, `num_references`,
`max_matching_error`, `landmark_scaling`, optional `reference_dir`). The repo ships
`knee_lateral` and `knee_axial`; any other request returns 400 with the available
keys. Add combinations by adding entries and their reference sets. If you mount this
file, editing it needs no rebuild. `GET /health` lists the current `supported` keys.

## Quick start (full stack: HTTPS, auth, MCP)

Build the image once, put your settings in a `service.env` next to `service.sh`,
then start it.

```bash
# 1. build (from the repo root)
docker build -f docker/Dockerfile -t roma_medical:latest .

# 2. configure: create service.env next to service.sh
cat > service.env <<'ENV'
API_KEY=choose-a-long-random-secret   # required on /process, /mcp, /upload
ROOT=/path/to/your/data-root          # holds model_weights/, refs/, the template
PUBLIC_IP=your.server.host.or.ip      # used in the TLS cert and printed URLs
GPUID=0
ENV

# 3. run
./service.sh start        # model + MCP + Caddy HTTPS, auto-restart
./service.sh status
./service.sh logs
./service.sh stop
```

This starts three containers on a private network: the model (localhost only), the
MCP server (internal), and Caddy, which terminates TLS on `HTTPS_PORT` (443 by
default) and routes by path. Clients then use these, with the `X-API-Key` header
(`/health` is open):

```
https://<PUBLIC_IP>/process    # REST
https://<PUBLIC_IP>/mcp         # MCP (Streamable HTTP)
```

Settings (env or `service.env`): `API_KEY`, `ROOT` (or `WEIGHTS_DIR`, `REFS_DIR`,
`TEMPLATE_FILE`), `PUBLIC_IP`, `GPUID`, `HTTPS_PORT`, `LOCAL_PORT`, `COARSE_RES`,
`UPSAMPLE_RES`.

### HTTPS certificate

`service.sh` generates a self-signed cert (with `PUBLIC_IP` in its SAN) on first
start, so clients trust it or skip verification (`curl -k`, `verify=False`). For a
CA-signed cert, put your `cert.pem` and `key.pem` in `<data-root>/caddy/`.

## REST service only (no HTTPS or MCP)

To front it with your own proxy, run just the model API:

```bash
docker run -d --name roma-medical --gpus all -p 8000:8000 --shm-size 8g \
  -e API_KEY=choose-a-long-random-secret \
  -v <data-root>/model_weights:/app/ThirdParty/model_weights:ro \
  -v <data-root>/refs:/refs:ro \
  -v <data-root>/experiment_config_template.json:/app/experiment_config_template.json:ro \
  roma_medical:latest
# http://localhost:8000/process (X-API-Key), /health, docs at /docs
```

`docker-compose.yml` is the same single service; set `WEIGHTS_DIR`, `REFS_DIR`,
`TEMPLATE_FILE`, and optionally `API_KEY`.

## Authentication

Set `API_KEY` to require the `X-API-Key` header on `/process`, `/mcp`, and
`/upload`. Leave it unset only for local use. `/health` stays open and reports
`auth_required`.

## REST API

Base URL is `https://<PUBLIC_IP>` (full stack) or `http://localhost:8000` (REST only).

```bash
curl -k https://<PUBLIC_IP>/health
# {"status":"ok","model_loaded":true,"auth_required":true,"supported":[...]}
```

`POST /process` (multipart form):

| Field | Required | Description |
|-------|----------|-------------|
| `image` | yes | radiograph (jpg/png) |
| `anatomy` | yes | e.g. `knee` |
| `projection` | yes | `lateral` or `axial` |
| `mpp` | no | mm per pixel; overrides the template default |
| `num_references` | no | references to match (`-1` = all); overrides the default |
| `max_matching_error` | no | max Procrustes error; overrides the default |

Omitted optional fields use the template default. `num_references=10` runs in about
15 s; the default matches all references (about 2 min).

```python
import requests
with open('target.jpg', 'rb') as f:
    r = requests.post('https://<PUBLIC_IP>/process',
                      headers={'X-API-Key': '<key>'},
                      files={'image': f},
                      data={'anatomy': 'knee', 'projection': 'lateral', 'num_references': 10},
                      verify=False)   # self-signed cert
r.raise_for_status()
print(r.json()['measurements'])
```

Response:

```json
{
  "anatomy": "knee", "projection": "lateral", "mpp": 0.148,
  "landmarks": [[x, y]],
  "measurements": {"<name>": 0.0},
  "matching": {"num_references_used": 10, "mean_confidence": 0.9}
}
```

A ready client is in [`client_example.py`](client_example.py):

```bash
python docker/client_example.py --url https://<PUBLIC_IP> --api-key <key> \
  --image target.jpg --anatomy knee --projection lateral
```

## MCP server (for agents)

A remote MCP server (Streamable HTTP) lets an agent call morphometry as a tool. It
returns a structured `status`, so the agent can tell why a call failed without extra
tools.

Point your MCP client at `https://<PUBLIC_IP>/mcp` with the `X-API-Key` header:

```json
{
  "mcpServers": {
    "roma-morphometry": {
      "url": "https://<PUBLIC_IP>/mcp",
      "headers": { "X-API-Key": "<key>" }
    }
  }
}
```

For the self-signed cert, trust it or disable verification (code clients: an httpx
factory with `verify=False`, shown below).

Images go in two steps, so pixel data stays out of the agent's context:

1. `POST https://<PUBLIC_IP>/upload` with the raw file bytes and the `X-API-Key`
   header returns `{"image_token": "<token>", "bytes": N}` (single use, 10 min).
2. Pass that token to the tool.

Tools:

- `get_morphometry(anatomy, projection, image_token, mpp?, num_references?)`.
  `image_token` (from `/upload`) is the only image input. It returns a dict with a
  `status` field:

  | `status` | meaning | also includes |
  |----------|---------|---------------|
  | `ok` | success | `measurements`, `landmarks`, `matching` |
  | `unsupported` | anatomy/projection not enabled | `supported`, `detail` |
  | `no_match` | no consensus (often wrong laterality/anatomy) | `detail` |
  | `unavailable` | service unreachable or not loaded | `detail` |
  | `bad_input` | bad or expired token, or bad arguments | `detail` |
  | `error` | unexpected | `detail` |

  The agent branches on `status`; no separate diagnostic tools are needed.
- `morphometry_health()` returns `{status, supported: [...]}`.

Full example (needs `mcp`, `httpx`, `requests`):

```python
import asyncio, json, requests, httpx, urllib3
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE, KEY = "https://<PUBLIC_IP>", "<key>"

def _factory(headers=None, timeout=None, auth=None):        # self-signed cert
    return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, verify=False)

def upload(path):
    r = requests.post(f"{BASE}/upload", headers={"X-API-Key": KEY},
                      data=open(path, "rb").read(), verify=False)
    r.raise_for_status()
    return r.json()["image_token"]

async def measure(path, anatomy, projection):
    token = upload(path)                                     # step 1
    async with streamablehttp_client(f"{BASE}/mcp", headers={"X-API-Key": KEY},
                                     httpx_client_factory=_factory) as (r, w, _):
        async with ClientSession(r, w) as s:                # step 2
            await s.initialize()
            res = await s.call_tool("get_morphometry",
                {"anatomy": anatomy, "projection": projection, "image_token": token})
            return json.loads(res.content[0].text)

print(asyncio.run(measure("target.jpg", "knee", "lateral")))
```

You can hand this section to a coding agent to wire the tool into your framework.

## Notes

- One GPU per container. Inference is serialized. For several GPUs, run one stack per
  GPU (set `GPUID` and distinct ports/names).
- Uploads are re-encoded to the reference set's image extension; lossless for
  grayscale radiographs.
- A `422` from `/process` means no consensus, usually a laterality or anatomy
  mismatch with the reference set.
- The image installs a curated dependency set (`requirements-docker.txt`), not the
  full dev `requirements.txt`.
- `do_preparation.py` is also a standalone CLI (and the hook for future DICOM support):
  ```bash
  python do_preparation.py --anatomy knee --projection lateral --mpp 0.148 \
    --image target.jpg --references_root /path/to/refs --job_dir ./job1
  ```
