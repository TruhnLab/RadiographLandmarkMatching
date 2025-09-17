#!/usr/bin/env bash

# Set environment variables
export PYTHONPATH="/app:/app/ThirdParty"
export MPLBACKEND=Agg
export MKL_SERVICE_FORCE_INTEL=1

echo "/app/mnt contents:"
ls -l /app/mnt
ls -l /app/mnt/knee_lateral/input/


python3 do_matching.py \
    --reference_path "/app/mnt/knee_lateral/references/*" \
    --data_path /app/mnt/knee_lateral/input \
    --save_path /app/mnt/knee_lateral/output \
    --left_reference_path /app/mnt/knee_lateral/reference_left/1010500000718410_9190787601/1010500000718410_9190787601_LATERAL_LEFT \
    --right_reference_path /app/mnt/knee_lateral/reference_right/1010500001799818_9190675901/1010500001799818_9190675901_LATERAL_LEFT \
    --image_filetype jpg \
    --max_matching_error 500

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "Processing completed successfully."
else
    echo "Processing failed with exit code $?"
    exit 1
fi
