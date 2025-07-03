from argparse import ArgumentParser
from pathlib import Path
import random

from acroface.experiment.tracker import ExperimentTracker, find_top_level_resamples
from acroface.experiment.core import ExperimentConfig, run_experiment, process_work_packages, find_pending_resamples, continue_experiments
from acroface.util import timestamp, load_object

def main():
    parser = ArgumentParser(description="Continue on experiment")
    parser.add_argument('experiment_root', help="Directory to recursively search for experiments", type=Path)
    args = parser.parse_args()
    config = load_object(args.experiment_config, ExperimentConfig)
    
    experiment_directory = args.experiment_root
    root_tracker = ExperimentTracker(experiment_directory)
    experiments = root_tracker.get_children_paths()
    
    continue_experiments(config, experiments)
    
        


if __name__ == '__main__':
    main()
