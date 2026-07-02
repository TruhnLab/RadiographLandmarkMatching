#!/usr/bin/env bash
#
# Manage the Roma Medical stack: model service + MCP server + Caddy HTTPS front.
# Clients reach it at:
#     https://<PUBLIC_IP>/process   (REST morphometry, X-API-Key header)
#     https://<PUBLIC_IP>/mcp       (MCP server, Streamable HTTP, X-API-Key header)
# Caddy terminates TLS on HTTPS_PORT (443 by default) and routes by path; the
# model and MCP containers stay internal (localhost / a private Docker network).
#
#   ./service.sh start | stop | restart | status | logs | health
#
# Configure via a `service.env` file next to this script (recommended) or env vars:
#   API_KEY=...           require this key on /process, /mcp, /upload (unset = no auth)
#   ROOT=/path/to/data    folder holding model_weights/, refs/, and the template json
#   PUBLIC_IP=host-or-ip  address clients use (goes into the TLS cert + printed URLs)
#   GPUID=0               which GPU to pin
# or point WEIGHTS_DIR / REFS_DIR / TEMPLATE_FILE at custom locations directly.
#
set -uo pipefail

# Load local config first, so it can set ROOT / PUBLIC_IP / GPUID / API_KEY / ...
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/service.env}"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi

IMAGE="${IMAGE:-roma_medical:latest}"
NAME="${NAME:-roma-medical}"
MCP_NAME="${MCP_NAME:-mcp-server}"
MCP_PORT="${MCP_PORT:-9000}"
CADDY_NAME="${CADDY_NAME:-caddy}"
NETWORK="${NETWORK:-roma-net}"
HTTPS_PORT="${HTTPS_PORT:-443}"
LOCAL_PORT="${LOCAL_PORT:-18080}"        # model, published on localhost only
GPUID="${GPUID:-0}"
ROOT="${ROOT:-$HOME/roma-medical}"
PUBLIC_IP="${PUBLIC_IP:-localhost}"
# Derived from ROOT (after the config file, so a ROOT override propagates here)
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/model_weights}"
REFS_DIR="${REFS_DIR:-$ROOT/refs}"
TEMPLATE_FILE="${TEMPLATE_FILE:-$ROOT/experiment_config_template.json}"
CADDY_DIR="${CADDY_DIR:-$ROOT/caddy}"

ensure_network() { docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null; }

ensure_caddy_config() {
  mkdir -p "$CADDY_DIR/data"
  if [ ! -f "$CADDY_DIR/cert.pem" ] || [ ! -f "$CADDY_DIR/key.pem" ]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
      -keyout "$CADDY_DIR/key.pem" -out "$CADDY_DIR/cert.pem" \
      -subj "/CN=roma-medical" \
      -addext "subjectAltName=IP:${PUBLIC_IP},IP:127.0.0.1,DNS:localhost" 2>/dev/null
    echo "generated self-signed cert for ${PUBLIC_IP}"
  fi
  # /mcp + /upload -> MCP server (with API-key gate if a key is set); else -> model REST
  local mcp_block
  if [ -n "${API_KEY:-}" ]; then
    mcp_block="    @mcpapi path /mcp* /upload*
    handle @mcpapi {
        @unauth not header X-API-Key ${API_KEY}
        respond @unauth \"unauthorized\" 401
        reverse_proxy ${MCP_NAME}:${MCP_PORT}
    }"
  else
    mcp_block="    @mcpapi path /mcp* /upload*
    handle @mcpapi {
        reverse_proxy ${MCP_NAME}:${MCP_PORT}
    }"
  fi
  cat > "$CADDY_DIR/Caddyfile" <<CF
{
    auto_https off
}
:443 {
    tls /etc/caddy/cert.pem /etc/caddy/key.pem
${mcp_block}
    handle {
        reverse_proxy ${NAME}:8000
    }
}
CF
}

start() {
  ensure_network
  ensure_caddy_config

  # --- model service: on the shared network, published on localhost only ---
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  local args=(-d --name "$NAME" --restart unless-stopped --network "$NETWORK"
    --gpus "\"device=${GPUID}\"" -p "127.0.0.1:${LOCAL_PORT}:8000" --shm-size 8g
    -v "${WEIGHTS_DIR}:/app/ThirdParty/model_weights:ro"
    -v "${REFS_DIR}:/refs:ro"
    -v "${TEMPLATE_FILE}:/app/experiment_config_template.json:ro")
  if [ -n "${API_KEY:-}" ]; then args+=(-e "API_KEY=${API_KEY}"); fi
  docker run "${args[@]}" "$IMAGE" >/dev/null

  # --- MCP server: forwards to the model over the private network ---
  docker rm -f "$MCP_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$MCP_NAME" --restart unless-stopped --network "$NETWORK" \
    -e "MORPH_URL=http://${NAME}:8000" -e "MORPH_API_KEY=${API_KEY:-}" \
    -e FASTMCP_HOST=0.0.0.0 -e "FASTMCP_PORT=${MCP_PORT}" \
    --entrypoint python3 "$IMAGE" /app/mcp_server.py >/dev/null

  # --- Caddy HTTPS front on :443 ---
  docker rm -f "$CADDY_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$CADDY_NAME" --restart unless-stopped --network "$NETWORK" -p "${HTTPS_PORT}:443" \
    -v "${CADDY_DIR}/Caddyfile:/etc/caddy/Caddyfile:ro" \
    -v "${CADDY_DIR}/cert.pem:/etc/caddy/cert.pem:ro" \
    -v "${CADDY_DIR}/key.pem:/etc/caddy/key.pem:ro" \
    -v "${CADDY_DIR}/data:/data" caddy:2 >/dev/null

  echo "started '$NAME' + '$MCP_NAME' + '$CADDY_NAME' (GPU ${GPUID}); loading model..."
  if [ -n "${API_KEY:-}" ]; then echo "auth: ON (X-API-Key required)"; else echo "auth: OFF (no API_KEY set)"; fi
  for _ in $(seq 1 48); do
    if curl -s "http://localhost:${LOCAL_PORT}/health" 2>/dev/null | grep -q '"model_loaded":true'; then
      echo "ready:"
      echo "   REST -> https://${PUBLIC_IP}/process"
      echo "   MCP  -> https://${PUBLIC_IP}/mcp"
      return 0
    fi
    sleep 5
  done
  echo "WARNING: model not ready after 4 min; check '$0 logs'"
  return 1
}

stop() {
  for c in "$CADDY_NAME" "$MCP_NAME" "$NAME"; do
    docker rm -f "$c" >/dev/null 2>&1 && echo "stopped '$c'" || true
  done
}
logs()    { docker logs -f "${1:-$NAME}"; }
status()  { docker ps -a --filter "name=$NAME" --filter "name=$MCP_NAME" --filter "name=$CADDY_NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'; }
health()  { curl -sk "https://localhost:${HTTPS_PORT}/health" && echo; }

case "${1:-}" in
  start)   start   ;;
  stop)    stop    ;;
  restart) stop; start ;;
  logs)    shift; logs "${1:-}" ;;
  status)  status  ;;
  health)  health  ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs [container]|health}"; exit 1 ;;
esac
