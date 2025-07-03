from argparse import ArgumentParser
from pathlib import Path
import random


from acroface.experiment.tracker import ExperimentTracker
from acroface.experiment.core import ExperimentConfig, run_experiment
from acroface.util import timestamp, load_object

def main():
    parser = ArgumentParser(description="Run experiment")
    parser.add_argument('experiment_config', help="Main experiment config file", type=Path)
    parser.add_argument('--output-dir', help="Root directory to write experiments to", default=Path('experiments'), type=Path)
    args = parser.parse_args()
    run_experiment(args.experiment_config, args.output_dir)


if __name__ == '__main__':
    main()
