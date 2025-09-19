# IMPORTS
import sys
sys.path.append('./ThirdParty')

import os
import time
import torch
import glob
import json
import numpy as np
from skimage import io
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from matplotlib import pyplot as plt

from ThirdParty.romatch import roma_outdoor
from utils.utils import read_csv, create_csv, load_model_weights, \
                        compute_matches, procrustes, geometric_median
from utils.vis2D import plot_images, plot_matches, save_plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(hparams):
    """
    Runtime analysis script for landmark matching process
    :param hparams:
    """

    # ------------------------
    # DATA DEFINITIONS
    # ------------------------

    # Check which files have already been processed
    os.makedirs(hparams.save_path, exist_ok=True)
    
    # Get all target images
    target_img_paths = glob.glob(os.path.join(hparams.data_path, f'*.{hparams.image_filetype}'))
    print(f'Found {len(target_img_paths)} target image files')

    # Get all reference files
    ref_img_files = sorted(glob.glob(os.path.join(hparams.reference_path, f'*_image.{hparams.image_filetype}')))
    ref_kpt_files = sorted(glob.glob(os.path.join(hparams.reference_path, '*_landmarks.csv')))
    assert len(ref_img_files) == len(ref_kpt_files), f"Mismatch between number of reference images and keypoint files. {len(ref_img_files)} images and {len(ref_kpt_files)} keypoint files found."
    count_ref_files = len(ref_img_files)
    print(f'Found {count_ref_files} reference files')

    # ------------------------
    # SET UP MODEL
    # ------------------------ 

    # Load weights directly to the correct device
    try:
        weights = load_model_weights("ThirdParty/model_weights/roma_outdoor.pth", device)
    except:
        print('No model weights found. Downloading default weights (internet required).')
        weights = None
    try: 
        dinov2_weights = load_model_weights("ThirdParty/model_weights/dinov2_vitl14_pretrain.pth", device)
    except:
        print('No model weights found. Downloading default weights (internet required).')
        dinov2_weights = None

    # Create the model with pre-loaded weights
    roma_model = roma_outdoor(device=device,
                              weights=weights,
                              dinov2_weights=dinov2_weights,
                              coarse_res=hparams.coarse_res,
                              upsample_res=hparams.upsample_res)
    roma_model.eval()
    try: # not yet working on windows (as of 06.25)
        roma_model = torch.compile(roma_model)
    except:
        pass

    # ------------------------
    # RUNTIME ANALYSIS
    # ------------------------
    
    matching_count = 0
    matching_times = []
    confidence_values = []
    procrustes_errors = []
    
    # Create results directory
    benchmark_dir = os.path.join(hparams.save_path, "benchmark_results")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    print(f"Starting benchmark - Running until {hparams.max_matchings} matchings are complete...")
    
    # Iterate through all target images until we reach max_matchings
    for target_img_path in target_img_paths:
        if matching_count >= hparams.max_matchings:
            break
            
        id_target = os.path.split(target_img_path)[-1].replace('.png','').replace('.jpg','')
        print(f"\nProcessing target image: {id_target}")
        
        # Create directory for individual matching results
        target_benchmark_dir = os.path.join(benchmark_dir, f"{id_target}_matches")
        os.makedirs(target_benchmark_dir, exist_ok=True)

        # MATCHING WITH REFERENCE FILES (No laterality check)
        for num_matching, (ref_img_path, ref_kpt_path) in enumerate(zip(ref_img_files, ref_kpt_files)):
            if matching_count >= hparams.max_matchings:
                break
                
            # Extract ID from file names
            id_img = os.path.split(ref_img_path)[-1].replace('_image.jpg', '').replace(f'_image.{hparams.image_filetype}', '')
            
            # Record timing information
            print(f"Match {matching_count+1}/{hparams.max_matchings}: {id_img} to {id_target}...")
            
            start_time = time.time()
            img_ref, img_target, keypoints_source, registered_kpts = compute_matches(
                ref_img_path, ref_kpt_path, target_img_path, roma_model, device, 
                landmark_scaling=hparams.landmark_scaling
            )
            elapsed_time = time.time() - start_time
            
            # Calculate confidence
            _, _, procrustes_error = procrustes(keypoints_source, registered_kpts)
            matching_confidence = 1 - np.clip(procrustes_error, 0, hparams.max_matching_error) / hparams.max_matching_error
            
            # Record metrics
            matching_times.append(elapsed_time)
            confidence_values.append(matching_confidence)
            procrustes_errors.append(procrustes_error)
            
            # Save detailed results
            match_data = {
                "matching_time": float(f"{elapsed_time:.4f}"),
                "matching_confidence": float(f"{matching_confidence:.4f}"),
                "procrustes_error": float(f"{procrustes_error:.4f}")
            }
            
            print(f"Match completed in {elapsed_time:.2f}s with confidence {matching_confidence:.2f}")
            
            # Save visualization if enabled
            if hparams.save_visualizations:
                save_path_image = os.path.join(target_benchmark_dir, f"{id_img}_to_{id_target}.png")
                save_path_metrics = os.path.join(target_benchmark_dir, f"{id_img}_to_{id_target}_metrics.json")
                
                plot_images([img_ref, img_target])
                plot_matches(keypoints_source, registered_kpts, color='red', lw=1.5, ps=6, a=0.5)
                plt.title(f'Time: {elapsed_time:.2f}s, Confidence: {matching_confidence:.2f}')
                save_plot(save_path_image)
                plt.close()
                
                with open(save_path_metrics, 'w') as f:
                    json.dump(match_data, f, indent=2)
            
            matching_count += 1
    
    # ------------------------
    # RESULTS ANALYSIS
    # ------------------------
    
    # Calculate statistics
    avg_time = np.mean(matching_times)
    std_time = np.std(matching_times)
    median_time = np.median(matching_times)
    min_time = np.min(matching_times)
    max_time = np.max(matching_times)
    
    avg_confidence = np.mean(confidence_values)
    avg_procrustes = np.mean(procrustes_errors)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Completed {matching_count} matchings")
    print(f"Average matching time: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Median matching time: {median_time:.4f}s")
    print(f"Min/Max matching time: {min_time:.4f}s / {max_time:.4f}s")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"Average Procrustes error: {avg_procrustes:.4f}")
    print("="*50)
    
    # Save aggregate results
    summary_data = {
        "num_matchings": matching_count,
        "device": str(device),
        "time_stats": {
            "average": float(f"{avg_time:.4f}"),
            "std_dev": float(f"{std_time:.4f}"),
            "median": float(f"{median_time:.4f}"),
            "min": float(f"{min_time:.4f}"),
            "max": float(f"{max_time:.4f}")
        },
        "confidence_stats": {
            "average": float(f"{avg_confidence:.4f}")
        },
        "procrustes_stats": {
            "average": float(f"{avg_procrustes:.4f}")
        },
        "individual_matchings": [
            {
                "matching_time": float(f"{t:.4f}"),
                "matching_confidence": float(f"{c:.4f}"),
                "procrustes_error": float(f"{e:.4f}")
            }
            for t, c, e in zip(matching_times, confidence_values, procrustes_errors)
        ]
    }
    
    with open(os.path.join(benchmark_dir, "benchmark_summary.json"), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Visualize time distribution
    if hparams.save_visualizations:
        plt.figure(figsize=(10, 6))
        plt.hist(matching_times, bins=20, alpha=0.7, color='blue')
        plt.axvline(avg_time, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_time:.4f}s')
        plt.axvline(median_time, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_time:.4f}s')
        plt.xlabel('Matching Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Matching Times')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(benchmark_dir, "time_distribution.png"))
        plt.close()
        
        # Plot confidence vs time
        plt.figure(figsize=(10, 6))
        plt.scatter(matching_times, confidence_values, alpha=0.6)
        plt.xlabel('Matching Time (seconds)')
        plt.ylabel('Matching Confidence')
        plt.title('Matching Time vs. Confidence')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(benchmark_dir, "time_vs_confidence.png"))
        plt.close()


if __name__ == '__main__':
    # ------------------------
    # BENCHMARK ARGUMENTS
    # ------------------------
    
    parent_parser = ArgumentParser(
        description='Roma Medical - Benchmark for landmark matching performance',
        epilog='Example: python benchmark_matching.py --data_path /path/to/images --reference_path /path/to/references',
        formatter_class=RawDescriptionHelpFormatter,
        add_help=True
    )

    parent_parser.add_argument(
        '--reference_path',
        type=str,
        default=r'E:\data\UKAFeetX\AP_LEFT\*',
        help='Path to reference images and landmark files'
    )

    parent_parser.add_argument(
        '--data_path',
        type=str,
        default=r'E:\data\UKAFeetX\AP_LEFT\*',
        help='Path to target images to be processed'
    )

    parent_parser.add_argument(
        '--save_path',
        type=str,
        default=r'E:\experiments\MSK_Landmarks_2D\benchmark',
        help='Path where benchmark results will be saved'
    )
    
    parent_parser.add_argument(
        '--image_filetype',
        type=str,
        default='jpg',
        help='Image file extension (jpg, png, etc.)'
    )
    
    parent_parser.add_argument(
        '--max_matchings',
        type=int,
        default=100,
        help='Number of matchings to perform for the benchmark'
    )
    
    parent_parser.add_argument(
        '--max_matching_error',
        type=int,
        default=500,
        help='Maximum allowed matching error threshold'
    )

    parent_parser.add_argument(
        '--coarse_res',
        type=int,
        default=(560,560),
        nargs='+',
        help='Coarse resolution for processing (width height)'
    )

    parent_parser.add_argument(
        '--upsample_res',
        type=int,
        default=(1120,1120),
        nargs='+',
        help='Upsample resolution for processing (width height)'
    )
    
    parent_parser.add_argument(
        '--landmark_scaling',
        type=int,
        default=(1,1),
        nargs='+',
        help='Landmark scaling factors (x_scale y_scale)'
    )
    
    parent_parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Save visualizations of matches and performance metrics'
    )

    hyperparams = parent_parser.parse_args()

    # ---------------------
    # RUN BENCHMARK
    # ---------------------
    main(hyperparams)
