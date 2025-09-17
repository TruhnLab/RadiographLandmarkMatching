# IMPORTS
import sys
sys.path.append('./ThirdParty')

import os
import glob
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from utils.utils import read_csv, load_config
from measurements.measurements import get_function_dict




def main(hparams):

    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # DATA DEFINITIONS
    # ------------------------

    # Get configurations
    config = load_config(hparams.config_tag, config_path=hparams.config_path)
    function_dict, _ = get_function_dict(mode=config['mode'])
    function_names = [key for key in function_dict.keys()]

    # Get all target files
    kpt_paths = glob.glob(os.path.join(hparams.data_path, '*.csv'))
    kpt_paths = [k for k in kpt_paths if '_matches_bulk.csv' in k]
    print('Found {0} keypoint files'.format(len(kpt_paths)))

    # Prepare measurment dict
    measure_df = pd.DataFrame(columns=['ID'] + function_names)


    # ------------------------
    # MEASUREMENT
    # ------------------------

    for kpt_path in kpt_paths:

        id_target = os.path.split(kpt_path)[-1].replace(f'_matches_bulk.csv','')
        new_measure = [id_target,]

        # Load data
        kpt_data = read_csv(kpt_path)
        kpt_data = np.array([[float(y)*config['mpp'],float(x)*config['mpp']] for x,y in kpt_data])

        # calculate measures
        for measure_key in function_dict.keys():
            measure_function = function_dict[measure_key][0]
            measure_value = measure_function(kpt_data)
            new_measure.append(float(measure_value))

        # Append to dataframe
        measure_df.loc[len(measure_df)] = new_measure


    # Save measurements
    measure_df.to_csv(os.path.join(hparams.save_path, f'measurements_{hparams.config_tag}.csv'), index=False, sep=';')





if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are deploy-related arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--data_path',
        type=str,
        default=r'E:\experiments\MSK_Landmarks_2D\ROMA_Knee_PatellaMeasures_2010-2024'
    )

    parent_parser.add_argument(
        '--save_path',
        type=str,
        default=r'E:\experiments\MSK_Landmarks_2D\ROMA_Knee_PatellaMeasures_2010-2024\measurements'
    )

    parent_parser.add_argument(
        '--config_path',
        type=str,
        default=r'C:\Users\deschweiler\Documents\KneeMRI_PatellofemoralMeasurements\roma_medical\experiment_config_windows.json'
    )

    parent_parser.add_argument(
        '--config_tag',
        type=str,
        default='knee_lateral'
    )

    
    hyperparams = parent_parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)