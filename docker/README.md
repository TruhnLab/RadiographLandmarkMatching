# Roma Medical - Docker Setup Guide (Production Compatible)

This guide will help you run the Roma Medical matching script using Docker with a production-compatible temporary directory workflow that matches your project's architecture.

## Overview

The Roma Medical Docker setup uses a **temporary directory pattern** for production compatibility:

- ✅ **Production Compatible**: Matches the overarching project's workflow
- ✅ **GPU/CPU Support**: Automatic detection with graceful fallback
- ✅ **No Fixed Directories**: Uses temporary directories for isolation
- ✅ **Flexible Integration**: Easy to integrate into existing systems
- ✅ **Self-Contained**: All dependencies included in Docker image

## Prerequisites

1. **Docker Desktop**
   - Windows: Download from [docker.com](https://www.docker.com/products/docker-desktop/)
   - Linux: Follow [official installation guide](https://docs.docker.com/engine/install/)
   - Restart after installation

2. **Python 3.x** (for production integration script)

3. **NVIDIA Docker Support** (Optional, for GPU acceleration)
   - Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - Requires NVIDIA GPU with recent drivers

4. **Verify Installation**
   ```bash
   docker --version
   nvidia-smi  # Optional: Check GPU status
   ```

## Quick Start

### Production Usage (Recommended)

Use the Python integration script for production workflows:

```python
from run_production import RomaMedicalDocker

# Initialize Docker runner
roma_docker = RomaMedicalDocker("roma_medical:latest")

# Run matching with automatic temporary directories
results = roma_docker.run_matching(
    data_path="/path/to/target/images",           # Folder containing images to analyze
    reference_path="/path/to/reference/files",   # Folder with reference images & landmarks
    reference_left_path="/path/to/left/ref",     # Base path for left laterality check
    reference_right_path="/path/to/right/ref",   # Base path for right laterality check
    output_dir="/path/to/save/results",          # Where to save results permanently
    max_matching_error=500,
    image_filetype="jpg"
)

print(f"Processing completed! Results in: {results['output_directory']}")
```

### Development/Testing

For development and testing:

```bash
# Build and run for development/testing
cd docker
docker-compose up --build
```

## Architecture

### Temporary Directory Flow
```
Host System:
  📁 Your Images: /any/path/to/images/
  📁 Your References: /any/path/to/references/
  📁 Your Output: /path/to/save/outputs/to
  
Docker Execution:
  📁 Temp Dir: /tmp/xyz123/
  ├── 📁 input/      (copied from your images)
  ├── 📁 output/     (results generated here)
  └── 📁 references/ (copied from your references)
```

### Key Benefits:
- **Isolation**: Each execution is completely isolated
- **Cleanup**: Automatic cleanup of temporary files
- **Scalability**: Multiple containers can run simultaneously
- **Production Ready**: Matches existing project patterns

## Parameters

All parameters from `do_matching.py` are supported:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_path` | Path to target images folder | Required |
| `reference_path` | Path to reference images and landmarks | Required |
| `reference_left_path` | Base path for left laterality check | "" (optional) |
| `reference_right_path` | Base path for right laterality check | "" (optional) |
| `max_matching_error` | Maximum allowed matching error | 500 |
| `image_filetype` | Image file extension | "jpg" |
| `output_dir` | Permanent results directory | None (temporary) |

## GPU vs CPU Performance

The container automatically detects and adapts:

- **🚀 With GPU**: Significantly faster processing (recommended for production)
- **🐌 Without GPU**: Full functionality, slower processing (fine for development)
- **🔄 Automatic**: No configuration needed - container adapts automatically

## Build Instructions (Required First Step)

**⚠️ Important**: You must build the Docker image before using it (either from Python or command line).

### Option 1: Using docker-compose (Development)
```bash
cd docker
docker-compose build
```

### Option 2: Direct build (Production)
```bash
# From the main project directory (roma_medical/)
docker build -f docker/Dockerfile -t roma_medical:latest .
```

**After building, the image can be used in multiple ways:**
- ✅ Python script (via `run_production.py`)
- ✅ Command line (via `docker run`)
- ✅ Docker compose (via `docker-compose up`)

## Usage After Building

## Production Deployment

### Minimal Host Requirements:
- Python 3.x environment
- Docker installed
- The `run_production.py` script

### Deployment Steps:
1. **Build once:** `docker build -f docker/Dockerfile -t roma_medical:latest .` (from main project directory)
2. **Deploy anywhere:** Copy image + integration script  
3. **Execute:** Use from Python scripts, command line, or compose (no rebuild needed)

### Example Production Integration:
```python
import logging
from run_production import RomaMedicalDocker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Process knee X-rays
roma_docker = RomaMedicalDocker("roma_medical:latest", logger)

try:
    results = roma_docker.run_matching(
        data_path="/production/data/incoming",
        reference_path="/production/references/knee_lateral",
        output_dir="/production/results/batch_001"
    )
    logger.info(f"Batch processed: {len(results['output_files'])} files")
except Exception as e:
    logger.error(f"Processing failed: {e}")
```

## Understanding Output

Results include:
- **📊 Individual Matches**: Visualization for each reference-to-target pair
- **📍 Landmark Coordinates**: CSV files with precise keypoint positions  
- **📈 Matching Metrics**: JSON files with confidence and error data
- **🎯 Consensus Results**: Bulk analysis showing agreement across references

## Troubleshooting

### Common Issues

**"Docker not found"**
- Ensure Docker Desktop is installed and running
- Restart computer after installation

**"Permission denied"**
- Run terminal/command prompt as Administrator
- Ensure Docker Desktop is running

**"Out of space"**
- Clean old Docker images: `docker system prune`
- Free up disk space

**"Container exits immediately"**
- Check logs: `docker-compose logs`
- Verify image and reference paths exist

**"GPU not detected"**
- Install NVIDIA Container Toolkit
- Verify with `nvidia-smi`
- Container will automatically fall back to CPU

### Debug Mode

For debugging:
```bash
# Interactive container access
docker-compose run roma_medical bash

# Check GPU detection
docker run --rm --runtime=nvidia --gpus all roma_medical:latest python -c "import torch; print(torch.cuda.is_available())"
```

## Advanced Usage

### Custom Docker Command
```bash
# Manual Docker execution with custom mount
docker run --rm --runtime=nvidia --gpus all \
  -v "/path/to/data:/app/mnt:rw" \
  roma_medical:latest \
  python do_matching.py --max_matching_error 300
```

### Integration with Existing Systems
The `run_production.py` script is designed to integrate seamlessly with existing workflows that use temporary directory patterns, making it compatible with other AI processing pipelines in your project.

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify file paths and permissions
3. Test with `docker --version` and `nvidia-smi`
4. Review container output for specific error messages

The Docker setup is designed to be production-ready while maintaining ease of use for development and testing.
