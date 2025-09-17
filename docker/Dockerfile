# Use CUDA base image for GPU support with CPU fallback capability
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set working directory inside the container
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Install core dependencies with explicit versions 
RUN pip3 install "numpy==1.24.3"
RUN pip3 install "scikit-image==0.21.0"
RUN pip3 install "matplotlib==3.7.2"
RUN pip3 install "opencv-python==4.8.0.76"

# Install PyTorch with CUDA support if available, fallback to CPU-only version
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu124 || \
    pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip3 install "pillow==10.0.0" "pathlib" "argparse"

# Create directory for data storage
RUN mkdir -p /app/mnt

# Copy necessary files and directories
COPY ./ThirdParty /app/ThirdParty
COPY ./utils /app/utils
COPY ./measurements /app/measurements
COPY ./experiment_config_windows.json /app
COPY ./do_matching.py /app
COPY ./do_measurements.py /app

# Set environment variables (compatible with production system)
ENV PYTHONPATH="/app:/app/ThirdParty"
ENV MPLBACKEND=Agg
ENV MKL_SERVICE_FORCE_INTEL=1

# Create entrypoint script
COPY docker/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Set permissions (compatible with production workflow)
RUN chown -R root:root /app && chmod -R 775 /app

# Use entrypoint for production compatibility
ENTRYPOINT ["/app/entrypoint.sh"]
