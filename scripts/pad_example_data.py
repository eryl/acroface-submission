import argparse
from collections import defaultdict
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description="Script for example use only, make multiple copies of data for illustration")
    parser.add_argument("dataset_root", help="Directory with the example data, should have the subdirectories 'training_dataset' and 'test_dataset'", type=Path)
    parser.add_argument("-n", help="How many subject pairs to pad up to", type=int, default=10)
    args = parser.parse_args()
    
    training_data_path = args.dataset_root / 'training_dataset'
    test_data_path = args.dataset_root / 'test_dataset'
    
    training_subdirs = sorted(training_data_path.iterdir())
    test_subdirs = sorted(test_data_path.iterdir())
    
    common_subdirs = set([p.name for p in training_subdirs]) & set([p.name for p in test_subdirs])
    n_existing = len(common_subdirs)

    existing_groups = defaultdict(list)
    for subdir in sorted(common_subdirs):
        m = re.match(r"Sora(\d+)-([A|K])-(.*)", subdir)  # We hardcode this to the example dataset so it wont work for any other
        if m is not None:
            group_id, label, suffix = m.groups()
            existing_groups[group_id].append((label, suffix, subdir))
    
    flattened_groups = sorted(existing_groups.items())
    n_groups = len(flattened_groups)
    
    for i in range(n_groups, args.n):
        target_group_idx = i+1
        source_group_idx = i % n_groups
        source_group_id, source_pair = flattened_groups[source_group_idx]
        for src_subject in source_pair:
            label, suffix, src_name = src_subject
            training_source_dir = training_data_path / src_name
            training_target_dir = training_data_path / f"Sora{target_group_idx:02}-{label}-{suffix}"
            training_target_dir.mkdir(exist_ok=True)
            for source_file in training_source_dir.iterdir():
                m = re.match(r"Sora\d+-([A|K])-(.*)", source_file.name)
                if m is not None:
                    file_label, file_suffix = m.groups()
                    target_file_name = f"Sora{target_group_idx}-{file_label}-{file_suffix}"
                    target_file = training_target_dir / target_file_name
                    shutil.copy(source_file, target_file)

            test_source_dir = test_data_path / src_name
            test_target_dir = test_data_path / f"Sora{target_group_idx:02}-{label}-{suffix}"
            test_target_dir.mkdir(exist_ok=True)
            for source_file in test_source_dir.iterdir():
                m = re.match(r"Sora\d+-([A|K])-(.*)", source_file.name)
                if m is not None:
                    file_label, file_suffix = m.groups()
                    target_file_name = f"Sora{target_group_idx}-{file_label}-{file_suffix}"
                    target_file = test_target_dir / target_file_name
                    shutil.copy(source_file, target_file)
        
    
    

if __name__ == '__main__':
    main()