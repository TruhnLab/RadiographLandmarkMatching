# IMPORTS
import sys
sys.path.append('./ThirdParty')

import os
import torch
import glob
import json
import numpy as np
from skimage import io
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from matplotlib import pyplot as plt

from ThirdParty.romatch import roma_outdoor
from utils.utils import read_csv, create_csv, load_model_weights,\
                        compute_matches, procrustes, geometric_median
from utils.vis2D import plot_images, plot_matches, save_plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(hparams):

    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # DATA DEFINITIONS
    # ------------------------

    # Check which files have already been processed
    os.makedirs(hparams.save_path, exist_ok=True)
    processed_paths = glob.glob(os.path.join(hparams.save_path, '*.csv'))

    # Get all target images
    target_img_paths = glob.glob(os.path.join(hparams.data_path, f'*.{hparams.image_filetype}'))
    print('Found {0} target image files'.format(len(target_img_paths)))

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
    # MATCHING
    # ------------------------

    for target_img_path in target_img_paths:

        id_target = os.path.split(target_img_path)[-1].replace('.png','').replace('.jpg','')
        processed_path = [i for i in processed_paths if os.path.split(i)[-1].startswith(f'{id_target}_')]
        if len(processed_path)>0:
            print(f'Skipping {id_target} as it has already been processed...')
            continue

        # LATERALITY CHECK
        # ------------------------
        if not hparams.reference_left_file is None and not hparams.reference_right_file is None:

            # Get reference left and right files
            ref_left_img_file = hparams.reference_left_file+f'_image.{hparams.image_filetype}'
            ref_left_kpt_file = hparams.reference_left_file+'_landmarks.csv'
            ref_right_img_file = hparams.reference_right_file+f'_image.{hparams.image_filetype}'
            ref_right_kpt_file = hparams.reference_right_file+'_landmarks.csv'

            print(f'Performing laterality and anatomy check on {id_target}...')
            _, img_target, ref_kpts_left, pred_kpts_left = compute_matches(ref_left_img_file, ref_left_kpt_file, target_img_path, roma_model, device, landmark_scaling=hparams.landmark_scaling)
            _, img_target, ref_kpts_right, pred_kpts_right = compute_matches(ref_right_img_file, ref_right_kpt_file, target_img_path, roma_model, device, landmark_scaling=hparams.landmark_scaling)

            _, _, error_left = procrustes(ref_kpts_left, pred_kpts_left)
            _, _, error_right = procrustes(ref_kpts_right, pred_kpts_right)

            print(f"Left Procrustes Error: {error_left:.4f}")
            print(f"Right Procrustes Error: {error_right:.4f}")

            if error_left > hparams.max_matching_error and error_right > hparams.max_matching_error: # TODO: optimize this threshold
                print(f"All checks failed for {id_target}. Skipping...")
                continue
            elif error_left < error_right:
                print('Detected wrong laterality. Flipping image horizontally to align with references.')
                img_target = np.flip(img_target, axis=1)
                io.imsave(target_img_path, img_target)
            else:
                print('Laterality and anatomy check passed.')

        else:
            print(f'Laterality and anatomy check skipped for {id_target}. Assuming correct laterality.')


        # MATCHING WITH REFERENCE FILES
        # ------------------------

        os.makedirs(os.path.join(hparams.save_path, f'{id_target}_matches'), exist_ok=True)
        for num_matching, (ref_img_path, ref_kpt_path) in enumerate(zip(ref_img_files, ref_kpt_files)):

            # Extract ID from file names
            id_img = os.path.split(ref_img_path)[-1].replace('_image.jpg', '')
            id_kpt = os.path.split(ref_kpt_path)[-1].replace('_landmarks.csv', '')

            assert id_img == id_kpt, f"ID mismatch: {id_img} != {id_kpt}"

            # Check if the image has already been processed
            save_path_image = os.path.join(hparams.save_path, f'{id_target}_matches', f'{id_img}_to_{id_target}.png')
            save_path_matches = os.path.join(hparams.save_path, f'{id_target}_matches', f'{id_img}_to_{id_target}_matches.csv')
            save_path_metrics = os.path.join(hparams.save_path, f'{id_target}_matches', f'{id_img}_to_{id_target}_metrics.json')
            if os.path.exists(save_path_image) and os.path.exists(save_path_matches) and os.path.exists(save_path_metrics):
                print(f"Skipping {save_path_image} as matches already exist.")
                continue

            # Matching
            print(f"{num_matching+1}/{count_ref_files} {id_img} to {id_target}...")
            img_ref, img_target, keypoints_source, registered_kpts = compute_matches(ref_img_path, ref_kpt_path, target_img_path, roma_model, device, landmark_scaling=hparams.landmark_scaling)

            # Calculate confidence
            _, _, procrustes_error = procrustes(keypoints_source, registered_kpts)
            matching_confidence = 1 - np.clip(procrustes_error, 0, hparams.max_matching_error) / hparams.max_matching_error
            match_data = {
                "matching_confidence": float(f"{matching_confidence:.4f}"),
                "procrustes_error": float(f"{procrustes_error:.4f}")
            }
            print(f"Matching done with a confidence of {matching_confidence:.2f}...")

            # Save results
            plot_images([img_ref, img_target])
            plot_matches(keypoints_source, registered_kpts, color='red', lw=1.5, ps=6, a=0.5)
            plt.title(f'Matching confidence: {matching_confidence:.2f} ({procrustes_error:.4f})')
            save_plot(save_path_image)
            plt.close()

            create_csv(registered_kpts.numpy(force=True).astype(int), save_path_matches.replace('.csv',''), test_split=0, val_split=0)
            
            with open(save_path_metrics, 'w') as f:
                json.dump(match_data, f, indent=2)

        
        # AGREEMENT CALCULATION
        # ------------------------

        # Get all prediction files
        prd_kpt_files_stacked = glob.glob(os.path.join(hparams.save_path, f'{id_target}_matches', '*_matches.csv'))
        print(f'{len(prd_kpt_files_stacked)} prediction files found for {id_target}.')

        # Load and stack all predicted keypoints
        prd_kpts_stacked = None
        for prd_kpt_file in prd_kpt_files_stacked:

            prd_kpts = read_csv(prd_kpt_file)
            prd_kpts = np.array([[float(x),float(y)] for x,y in prd_kpts])

            if prd_kpts_stacked is None:
                prd_kpts_stacked = prd_kpts[None,...]
            else:
                prd_kpts_stacked = np.concatenate([prd_kpts_stacked, prd_kpts[None,...]], axis=0)

        # Calculate mean position
        prd_kpts_mean = np.zeros((prd_kpts_stacked.shape[1], 2))
        for i in range(prd_kpts_stacked.shape[1]):
            prd_kpts_mean[i] = geometric_median(prd_kpts_stacked[:, i, :])
            
        # Reshape keypoint list as bulk (for visualization)
        prd_kpts_bulk = np.reshape(prd_kpts_stacked, (-1,2))

        # Visualize and save
        save_path_bulk = os.path.join(hparams.save_path, f'{id_target}_matches_bulk.png')
        figsize = (img_target.shape[0] / 100, img_target.shape[1] / 100)  # Convert pixels to inches at 100 DPI
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.imshow(img_target, cmap='gray')
        ax.scatter(prd_kpts_bulk[:, 0], prd_kpts_bulk[:, 1], c='green', s=10, alpha=0.05)
        ax.scatter(prd_kpts_mean[:, 0], prd_kpts_mean[:, 1], c='red', s=40)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path_bulk, bbox_inches="tight", pad_inches=0)

        create_csv(prd_kpts_mean, save_path_bulk.replace('.png','').replace('.jpg',''), test_split=0, val_split=0)





if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are deploy-related arguments

    parent_parser = ArgumentParser(
        description='Roma Medical - AI-powered knee X-ray landmark matching and analysis',
        epilog='Example: python do_matching.py --data_path /path/to/images --reference_path /path/to/references',
        formatter_class=RawDescriptionHelpFormatter,
        add_help=True
    )

    parent_parser.add_argument(
        '--reference_path',
        type=str,
        default=r'E:\data\UKAKneeX\LATERAL_ALL\*',
        help='Path to reference images and landmark files'
    )

    parent_parser.add_argument(
        '--data_path',
        type=str,
        default=r'E:\data\UKAKneeX\LATERAL_PACS\*',
        help='Path to target images to be processed'
    )

    parent_parser.add_argument(
        '--save_path',
        type=str,
        default=r'E:\experiments\MSK_Landmarks_2D\docker_test',
        help='Path where results will be saved'
    )

    parent_parser.add_argument(
        '--reference_left_file',
        type=str,
        default=r'E:\data\UKAKneeX\LATERAL_LEFT\1010500000718410_9190787601\1010500000718410_9190787601_LATERAL_LEFT',
        help='Reference file for left laterality check (optional)'
    )
    
    parent_parser.add_argument(
        '--reference_right_file',
        type=str,
        default=r'E:\data\UKAKneeX\LATERAL_RIGHT\1010500001799818_9190675901\1010500001799818_9190675901_LATERAL_RIGHT',
        help='Reference file for right laterality check (optional)'
    )
    
    parent_parser.add_argument(
        '--image_filetype',
        type=str,
        default='jpg',
        help='Image file extension (jpg, png, etc.)'
    )
    
    parent_parser.add_argument(
        '--knn',
        type=int,
        default=15,
        help='Number of nearest neighbors for matching'
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

    
    hyperparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)