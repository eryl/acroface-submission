from argparse import ArgumentParser
from pathlib import Path
import json
import random


from acroface.experiment.tracker import ExperimentTracker
from acroface.experiment.core import ExperimentConfig, setup_data_splits
from acroface.util import timestamp, load_object

def main():
    parser = ArgumentParser(description="Run experiment")
    parser.add_argument('experiment_config', help="Main experiment config file", type=Path)
    args = parser.parse_args()

    dataset_splits = setup_data_splits(args.experiment_config)

    

if __name__ == '__main__':
    main()