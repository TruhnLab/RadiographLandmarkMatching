import os
import pydicom
import numpy as np
from PIL import Image
import argparse
import sys
import platform


def dicom_to_jpg(data_root, save_path, verbose=True):
    """
    Convert DICOM files to JPG format.
    
    :param data_root: Root directory containing DICOM files.
    :param save_path: Directory where converted JPG files will be saved.
    """
        
    os.makedirs(save_path, exist_ok=True)

    # Use UNC paths to handle Windows path length limitations (Windows only)
    is_windows = platform.system() == 'Windows'
    if is_windows:
        unc_data_root = r'\\?\{0}'.format(data_root)
    else:
        unc_data_root = data_root
    
    dcm_files = []
    for root, dirs, files in os.walk(unc_data_root):
            for file in files:
                if file.lower().endswith('.dcm'):
                    file_path = os.path.join(root, file)
                    # Convert back to normal path for further processing (Windows only)
                    if is_windows:
                        file_path = file_path.replace('\\\\?\\', '')
                    dcm_files.append(file_path)

    if verbose:
        print(f"Found {len(dcm_files)} DICOM files to process")

    saved_count = 0
    for num_file, dicom_file in enumerate(dcm_files):

        info = dicom_file.replace(data_root, '').split(os.path.sep)

        folder = info[1]
        label = info[-2]
        file_name = f'{folder}_{label}_{info[-1].replace(".dcm", ".jpg")}'

        try:
            if is_windows:
                dicom_data = pydicom.dcmread(r'\\?\{0}'.format(dicom_file))
            else:
                dicom_data = pydicom.dcmread(dicom_file)
        except (pydicom.errors.InvalidDicomError,FileNotFoundError):
            if verbose: print(f"Invalid DICOM file: {dicom_file}")
            continue
        try:
            dicom_image = dicom_data.pixel_array.astype(np.float32)
            dicom_image -= np.min(dicom_image)
            dicom_image /= np.max(dicom_image)
            dicom_image *= 255.0
            dicom_image = dicom_image.astype(np.uint8)
            dicom_image = Image.fromarray(dicom_image)
            dicom_image.save(os.path.join(save_path, file_name))
            if verbose: print(f'Saved {file_name} ({num_file+1}/{len(dcm_files)})')
            saved_count += 1
        except Exception as e:
            if verbose: print(f"Error processing {dicom_file}: {e}")
            continue
    if verbose: print(f"Saved {saved_count} JPG files from DICOM files.")


def main():
    """
    Main function to handle command line arguments and execute DICOM to JPG conversion.
    """
    parser = argparse.ArgumentParser(
        description='Convert DICOM files to JPG format'
    )
    
    parser.add_argument(
        '--data_root', '-d',
        type=str,
        required=True,
        help='Root directory containing DICOM files to convert'
    )
    
    parser.add_argument(
        '--save_path', '-s',
        type=str,
        required=True,
        help='Directory where converted JPG files will be saved'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.data_root):
        print(f"Error: Input directory '{args.data_root}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(args.data_root):
        print(f"Error: Input path '{args.data_root}' is not a directory.")
        sys.exit(1)
    
    # Print configuration
    if args.verbose:
        print("DICOM to JPG Converter")
        print("=" * 50)
        print(f"Data root:        {args.data_root}")
        print(f"Save path:        {args.save_path}")
        print(f"Verbose mode:     {args.verbose}")
        print("=" * 50)
    
    try:
        # Execute conversion
        dicom_to_jpg(args.data_root, args.save_path, verbose=args.verbose)
        
        if args.verbose:
            print("\nConversion completed successfully!")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


