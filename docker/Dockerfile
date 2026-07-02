# Roma Medical - resident inference service image.
# Build from the PROJECT ROOT:  docker build -f docker/Dockerfile -t roma_medical:latest .

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python + a couple of shared libs opencv/matplotlib expect at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && pip3 install --no-cache-dir --upgrade pip

WORKDIR /app

# PyTorch with CUDA 12.1 wheels (bundles its own CUDA runtime libraries).
# cu121 stays safely under the host driver's CUDA 12.2 ceiling.
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Curated runtime dependencies (cached separately from the source copy)
COPY docker/requirements-docker.txt /app/requirements-docker.txt
RUN pip3 install --no-cache-dir -r /app/requirements-docker.txt

# Application code. Model weights are NOT copied here: they are mounted at
# runtime into /app/ThirdParty/model_weights (see docker-compose.yml).
COPY ThirdParty /app/ThirdParty
COPY utils /app/utils
COPY measurements /app/measurements
COPY experiment_config_template.json /app/
COPY do_preparation.py /app/
COPY do_matching.py /app/
COPY do_measurements.py /app/
COPY docker/serve.py /app/serve.py
COPY docker/mcp_server.py /app/mcp_server.py
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH="/app:/app/ThirdParty" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MKL_SERVICE_FORCE_INTEL=1 \
    REFERENCES_ROOT=/refs \
    TEMPLATE_PATH=/app/experiment_config_template.json

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
