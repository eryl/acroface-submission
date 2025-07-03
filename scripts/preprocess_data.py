import argparse
from pathlib import Path

from acroface.dataset.preprocess import PreprocessingConfig, preprocess_dataset
from acroface.util import load_object

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_directory', help="Root dataset for videos", type=Path)
    parser.add_argument('config_path', help="Path to preprocessing config file", type=Path)
    args = parser.parse_args()
    
    preprocessing_config = load_object(args.config_path, PreprocessingConfig)
    preprocess_dataset(args.dataset_directory, preprocessing_config)
    

if __name__ == '__main__':
    main()