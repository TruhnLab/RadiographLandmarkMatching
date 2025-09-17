# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:55:13 2020

@author: Nutzer
"""

import os
import glob
import csv
import json
import torch
import numpy as np

from skimage import io
from scipy.spatial import KDTree
from skimage.transform import resize


def get_files(folders, data_root='', descriptor='', filetype='tif'):
    
    filelist = []
    
    for folder in folders:
        files = glob.glob(os.path.join(data_root, folder, '*'+descriptor+'*.'+filetype))
        filelist.extend([os.path.join(folder, os.path.split(f)[-1]) for f in files])
        
    return filelist
        
    
        
def read_csv(list_path, data_root=''):
    
    filelist = []
    
    with open(list_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row)==0 or np.sum([len(r) for r in row])==0: continue
            row = [os.path.join(data_root, r) for r in row]
            filelist.append(row)
            
    return filelist
   


def create_csv(data_list, save_path='list_folder/experiment_name', test_split=0.2, val_split=0.1, shuffle=False):
        
    if shuffle:
        np.random.shuffle(data_list)
    
    # Get number of files for each split
    num_files = len(data_list)
    num_test_files = int(test_split*num_files)
    num_val_files = int((num_files-num_test_files)*val_split)
    num_train_files = num_files - num_test_files - num_val_files
    
    # Adjust file identifier if there is no split
    if test_split>0 or val_split>0:
        train_identifier='_train.csv'
    else:
        train_identifier='.csv'
    
    # Get file indices
    file_idx = np.arange(num_files)
    
    # Save csv files
    if num_test_files > 0:
        test_idx = sorted(np.random.choice(file_idx, size=num_test_files, replace=False))
        with open(save_path+'_test.csv', 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in test_idx:
                writer.writerow(data_list[idx])
    else:
        test_idx = []
        
    if num_val_files > 0:
        val_idx = sorted(np.random.choice(list(set(file_idx)-set(test_idx)), size=num_val_files, replace=False))
        with open(save_path+'_val.csv', 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in val_idx:
                writer.writerow(data_list[idx])
    else:
        val_idx = []
    
    if num_train_files > 0:
        train_idx = sorted(list(set(file_idx) - set(test_idx) - set(val_idx)))
        with open(save_path+train_identifier, 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in train_idx:
                writer.writerow(data_list[idx])
            



def resize_images(data_dir, target_size=(512,512), data_type='png'):

    filelist = glob.glob(os.path.join(data_dir, '*'+data_type))

    for file in filelist:

        img = io.imread(file)
        img = resize(img, target_size)

        if img.max() <= 1:
            img = (255*img).astype(np.uint8)

        save_dir, save_name = os.path.split(file)
        save_name = save_name.split('.')
        io.imsave(os.path.join(save_dir, save_name[0]+'_rescaled.'+save_name[1]), img)



def load_config(config_tag, config_path='config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    if config_tag in config:
        return config[config_tag]
    else:
        raise ValueError(f"Use case '{config_tag}' not found in the config file.")
    

# Load pre-downloaded weights
def load_model_weights(weights_path, device):
    """Load model weights from a local file to the specified device"""
    return torch.load(weights_path, map_location=device)
    


def calculate_confidence_from_local_cluster(kpts1, kpts2, radius=10, cutoff=50):

    # Calculate local clusters in reference image
    kdt = KDTree(kpts1)
    local_clusters_kpts1 = kdt.query_ball_point(kpts1, radius)

    disagreements = []
    for local_cluster in local_clusters_kpts1:

        # Get local cluster in both reference and target images
        local_kpts1 = kpts1[local_cluster]
        local_kpts2 = kpts2[local_cluster]

        # Calculate transform paths from one cluster to the other
        # If there is a large variance in the reference paths, the match has likely low confidence
        paths = local_kpts1 - local_kpts2
        disagreement = np.add(*np.var(paths.numpy(force=True), axis=0))
        disagreements.append(disagreement)

    # Clip spreads and normalize to [0,1]
    spreads = np.clip(disagreements, 0, cutoff)
    spreads /= cutoff

    return 1-spreads



def geometric_median(points, eps=1e-5):

    """
    Compute the geometric median of a set of points in 2D.
    Parameters:
        points: np.ndarray of shape (N, 2) - input points
        eps: float - convergence threshold
    Returns:
        y: np.ndarray of shape (2,) - geometric median of the points
    """

    y = np.mean(points, axis=0)  # Initial guess
    while True:
        distances = np.linalg.norm(points - y, axis=1)
        non_zero = (distances != 0)
        if not np.any(non_zero):
            return y.astype(int)
        
        inv_distances = 1 / distances[non_zero]
        inv_distances /= inv_distances.sum()
        y1 = (points[non_zero] * inv_distances[:, np.newaxis]).sum(axis=0)
        
        if np.linalg.norm(y - y1) < eps:
            return y1.astype(int)
        y = y1



def compute_matches(ref_img_path, ref_kpt_path, im2_path, model, device, landmark_scaling=(1, 1)):

    """
    Computes matches between two images using the ROMA model.
    Parameters:
    - ref_img_path: Path to the reference image.
    - ref_kpt_path: Path to the reference keypoints CSV file.
    - im2_path: Path to the target image.
    - model: The model instance used for matching.
    - device: The device (CPU or GPU) on which the model runs.
    - landmark_scaling: Scaling factors for the keypoints (default is (1, 1)).
    Returns:
    - im1: The reference image.
    - im2: The target image.
    - keypoints_source: Keypoints from the reference image.
    - registered_kpts: Keypoints from the reference image registered to the target image.
    """

    #########################################
    # Data loading and preparation
    im1 = io.imread(ref_img_path)
    im2 = io.imread(im2_path)

    W_A, H_A = im1.shape[:2]
    W_B, H_B = im2.shape[:2]

    keypoints_source = read_csv(ref_kpt_path)
    keypoints_source = np.array([[float(x)*landmark_scaling[1],\
                                  float(y)*landmark_scaling[0]] for x,y in keypoints_source])
    keypoints_source = torch.tensor(keypoints_source)


    #########################################
    # Match images
    with torch.no_grad():
        warp, certainty = model.match(ref_img_path, im2_path, device=device)


    #########################################
    # Sample matches for estimation
    matches, certainty = model.sample(warp, certainty)
    kpts1, kpts2 = model.to_pixel_coordinates(matches, W_A, H_A, W_B, H_B)
    kpts1 = kpts1.cpu()
    kpts2 = kpts2.cpu()

    # Register keypoints
    registered_kpts = torch.zeros_like(keypoints_source)

    for num_kpt, source_kpt in enumerate(keypoints_source):

        # get closets grid points
        dist = torch.norm(kpts1-source_kpt, dim=1, p=None)
        knn = dist.topk(15, largest=False).indices
        closest_source_kpts = kpts1[knn,:]
        closest_target_kpts = kpts2[knn,:]

        # translate source points by average distance from source to target
        distances = closest_source_kpts - closest_target_kpts
        translation = torch.mean(distances, dim=0)
        target_kpt = source_kpt - translation

        # save current keypoint
        registered_kpts[num_kpt,:] = target_kpt


    return im1, im2, keypoints_source, registered_kpts



def procrustes(ref_kpts, prd_kpts):
    """
    Align shape prd_kpts to shape ref_kpts using Procrustes analprd_kptssis.
    
    Parameters:
        ref_kpts: np.ndarray of shape (N, 2) - reference shape
        prd_kpts: np.ndarray of shape (N, 2) - shape to align

    Returns:
        prd_kpts_aligned: np.ndarraprd_kpts of shape (N, 2) - aligned version of prd_kpts
        transform: dict with translation, rotation matriref_kpts, and scale
        error: float - Procrustes distance
    """

    # Center shapes
    ref_kpts_mean = ref_kpts.mean(axis=0)
    prd_kpts_mean = prd_kpts.mean(axis=0)
    ref_kpts_centered = ref_kpts - ref_kpts_mean
    prd_kpts_centered = prd_kpts - prd_kpts_mean

    # Normalize (scale to unit Frobenius norm)
    norm_ref_kpts = np.linalg.norm(ref_kpts_centered)
    norm_prd_kpts = np.linalg.norm(prd_kpts_centered)
    ref_kpts_scaled = ref_kpts_centered / norm_ref_kpts
    prd_kpts_scaled = prd_kpts_centered / norm_prd_kpts

    # Optimal rotation using SVD
    U, S, Vt = np.linalg.svd(ref_kpts_scaled.T @ prd_kpts_scaled)
    R = U @ Vt

    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Step 4: Applprd_kpts transformation
    prd_kpts_rotated = prd_kpts_scaled @ R.T
    prd_kpts_aligned = prd_kpts_rotated * norm_ref_kpts + ref_kpts_mean

    # Calculate difference (Procrustes distance)
    error = np.linalg.norm(ref_kpts - prd_kpts_aligned)

    transform = {
        'rotation': R,
        'scale': norm_ref_kpts / norm_prd_kpts,
        'translation': ref_kpts_mean - (prd_kpts_mean @ R.T * norm_ref_kpts / norm_prd_kpts)
    }

    return prd_kpts_aligned, transform, error