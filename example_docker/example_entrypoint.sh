#!/usr/bin/env bash

export nnUNet_raw="/app/nnUNet_raw_data"
export nnUNet_preprocessed="/app/"
export nnUNet_results="/app/nnUNet_results"

export MKL_SERVICE_FORCE_INTEL=1

echo "/app/mnt contents:"
ls -l /app/mnt

ls -l /app/mnt/hip/input/
ls -l /app/mnt/knee/input/
ls -l /app/mnt/ankle/input/

nnUNetv2_predict -i /app/mnt/hip/input -o /app/mnt/hip/output -d 8 -c 3d_fullres -f all -chk checkpoint_best.pth -device cuda
nnUNetv2_predict -i /app/mnt/knee/input -o /app/mnt/knee/output -d 21 -c 3d_fullres -f all -chk checkpoint_best.pth -device cuda
nnUNetv2_predict -i /app/mnt/ankle/input -o /app/mnt/ankle/output -d 22 -c 3d_fullres -f all -chk checkpoint_best.pth -device cuda

echo "Prediction completed for hip, knee, and ankle datasets."