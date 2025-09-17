#!/usr/bin/env python3
"""
Roma Medical Docker Call Example
Similar to nnU-Net pattern but adapted for Roma Medical knee X-ray landmark matching
"""

import os
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path

class RomaMedicalDockerCall:
    def __init__(self, docker_image="roma_medical:latest", logger=None):
        self.docker_image = docker_image
        self.logger = logger or logging.getLogger(__name__)

    def process_knee_images(self, target_images, reference_images, reference_left_dir, reference_right_dir, max_matching_error=500, image_filetype="jpg"):
        """
        Process knee X-ray images using Roma Medical Docker container
        
        Args:
            target_images: List of image file paths to process
            reference_images: List of reference image and landmark file pairs
            reference_left_dir: Path to left reference directory
            reference_right_dir: Path to right reference directory
            max_matching_error: Maximum matching error threshold
            image_filetype: Image file extension
            
        Returns:
            Dictionary with processing results
        """
        
        with tempfile.TemporaryDirectory() as tempdir:
            # Create directory structure (matching entrypoint expectations)
            Path(f'{tempdir}/knee_lateral/input').mkdir(parents=True)
            Path(f'{tempdir}/knee_lateral/output').mkdir(parents=True)
            Path(f'{tempdir}/knee_lateral/references').mkdir(parents=True)
            Path(f'{tempdir}/knee_lateral/reference_left').mkdir(parents=True)
            Path(f'{tempdir}/knee_lateral/reference_right').mkdir(parents=True)
            
            self.logger.info("Created temporary directory structure")
            
            # Copy target images to input directory
            for i, image_path in enumerate(target_images):
                if os.path.exists(image_path):
                    dest_name = f"target_{i:04d}.{image_filetype}"
                    shutil.copy2(image_path, f'{tempdir}/knee_lateral/input/{dest_name}')
                    self.logger.debug(f"Copied target image: {image_path} -> {dest_name}")
                else:
                    self.logger.warning(f"Target image not found: {image_path}")
            
            # Copy reference images and landmarks to references directory
            for ref_data in reference_images:
                if isinstance(ref_data, dict):
                    # Expecting {'image': path, 'landmarks': path, 'name': identifier}
                    image_path = ref_data.get('image')
                    landmarks_path = ref_data.get('landmarks')
                    ref_name = ref_data.get('name', f'ref_{len(os.listdir(f"{tempdir}/knee_lateral/references"))}')
                    
                    if image_path and os.path.exists(image_path):
                        shutil.copy2(image_path, f'{tempdir}/knee_lateral/references/{ref_name}_image.{image_filetype}')
                        self.logger.debug(f"Copied reference image: {ref_name}")
                    
                    if landmarks_path and os.path.exists(landmarks_path):
                        shutil.copy2(landmarks_path, f'{tempdir}/knee_lateral/references/{ref_name}_landmarks.csv')
                        self.logger.debug(f"Copied reference landmarks: {ref_name}")
                else:
                    # Expecting directory path
                    if os.path.isdir(ref_data):
                        ref_name = os.path.basename(ref_data)
                        dest_dir = Path(f'{tempdir}/knee_lateral/references/{ref_name}')
                        shutil.copytree(ref_data, dest_dir)
                        self.logger.debug(f"Copied reference directory: {ref_name}")
            
            # Copy left reference directory
            if reference_left_dir and os.path.exists(reference_left_dir):
                if os.path.isdir(reference_left_dir):
                    # Copy the specific subdirectory structure expected by entrypoint
                    parent_name = os.path.basename(os.path.dirname(reference_left_dir))
                    ref_name = os.path.basename(reference_left_dir)
                    dest_path = Path(f'{tempdir}/knee_lateral/reference_left/{parent_name}')
                    dest_path.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(reference_left_dir, dest_path / ref_name)
                    self.logger.debug(f"Copied left reference: {reference_left_dir}")
            
            # Copy right reference directory
            if reference_right_dir and os.path.exists(reference_right_dir):
                if os.path.isdir(reference_right_dir):
                    # Copy the specific subdirectory structure expected by entrypoint
                    parent_name = os.path.basename(os.path.dirname(reference_right_dir))
                    ref_name = os.path.basename(reference_right_dir)
                    dest_path = Path(f'{tempdir}/knee_lateral/reference_right/{parent_name}')
                    dest_path.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(reference_right_dir, dest_path / ref_name)
                    self.logger.debug(f"Copied right reference: {reference_right_dir}")
            
            # Build Docker command (similar to nnU-Net pattern)
            docker_cmd = [
                'docker',
                'run',
                '--rm',
                '--runtime=nvidia',
                '--gpus', 'all',
                '--shm-size', '32G',
                '--group-add', 'root',
                '-v', f'{tempdir}:/app/mnt:rw,Z',
                self.docker_image
            ]
            
            # CPU fallback command
            docker_cmd_cpu = [
                'docker',
                'run',
                '--rm',
                '--shm-size', '32G',
                '--group-add', 'root',
                '-v', f'{tempdir}:/app/mnt:rw,Z',
                self.docker_image
            ]
            
            self.logger.info("Running Roma Medical Docker container...")
            self.logger.debug(f"Docker command: {' '.join(docker_cmd)}")
            
            # Execute container (with GPU first, CPU fallback)
            proc = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            
            if proc.returncode == 125:  # Docker runtime error (likely GPU issue)
                self.logger.warning("GPU execution failed, falling back to CPU-only mode...")
                proc = subprocess.run(
                    docker_cmd_cpu,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
            
            if proc.returncode != 0:
                self.logger.error(f"Container exited with code {proc.returncode}. Logs: {proc.stdout.decode('utf-8')}")
                raise RuntimeError(f"Roma Medical processing failed with exit code {proc.returncode}.")
            
            self.logger.info(f"Container exited with code {proc.returncode}.")
            self.logger.debug(f'Container logs: {proc.stdout.decode("utf-8")}')
            
            # Collect output files
            output_files = []
            output_dir = Path(f'{tempdir}/knee_lateral/output')
            if output_dir.exists():
                for output_file in output_dir.rglob('*'):
                    if output_file.is_file():
                        output_files.append(str(output_file))
                        self.logger.debug(f"Found output file: {output_file.name}")
            
            # Return results (similar to nnU-Net pattern)
            results = {
                'return_code': proc.returncode,
                'logs': proc.stdout.decode('utf-8'),
                'output_files': output_files,
                'temp_output_directory': str(output_dir)
            }
            
            return results


def example_usage():
    """Example usage similar to the nnU-Net pattern"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize the Roma Medical Docker caller
    roma_caller = RomaMedicalDockerCall("roma_medical:latest", logger)
    
    # Define input data (adapt these paths to your actual data)
    target_images = [
        r'E:\data\UKAKneeX\sample\sample_img\0000061396\0000061396_LATERAL_PACS_image.jpg'
        # Add more target images as needed
    ]
    
    # Reference data - can be directories or individual files
    reference_images = [
        r'E:\data\UKAKneeX\sample\sample_ref\1000000000392774_9192333801_LATERAL_LEFT',
        r'E:\data\UKAKneeX\sample\sample_ref\1000000000479951_9190938402_LATERAL_LEFT',
        r'E:\data\UKAKneeX\sample\sample_ref\1000000001028788_9191005801_LATERAL_RIGHT'
        # Add more references as needed
    ]
    
    # Left and right reference directories (for laterality check)
    reference_left_dir = r'E:\data\UKAKneeX\LATERAL_LEFT\1010500000718410_9190787601\1010500000718410_9190787601_LATERAL_LEFT'
    reference_right_dir = r'E:\data\UKAKneeX\LATERAL_RIGHT\1010500001799818_9190675901\1010500001799818_9190675901_LATERAL_RIGHT'
    
    try:
        # Process the images
        results = roma_caller.process_knee_images(
            target_images=target_images,
            reference_images=reference_images,
            reference_left_dir=reference_left_dir,
            reference_right_dir=reference_right_dir,
            max_matching_error=500,
            image_filetype="jpg"
        )
        
        logger.info("Processing completed successfully!")
        logger.info(f"Return code: {results['return_code']}")
        logger.info(f"Number of output files: {len(results['output_files'])}")
        
        # Process results (similar to nnU-Net pattern of loading results)
        for output_file in results['output_files']:
            if output_file.endswith('.csv'):
                logger.info(f"Found CSV result: {os.path.basename(output_file)}")
                # You could load and process CSV files here
            elif output_file.endswith('.png'):
                logger.info(f"Found visualization: {os.path.basename(output_file)}")
                # You could process visualization files here
        
        logger.debug(f"Container logs:\n{results['logs']}")
        
    except RuntimeError as e:
        logger.error(f"Processing failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_usage()
