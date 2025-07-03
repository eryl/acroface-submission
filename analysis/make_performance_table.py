import argparse
import json
from pathlib import Path
import re
from collections import defaultdict

from sklearn.metrics import roc_auc_score, roc_curve


import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('bootstrap_results_file', help="Directory containing the prediction files", type=Path)
    parser.add_argument('--output-directory', help="Where to write bootstrap results", type=Path, default=Path("analysis") / "results_tables")
    args = parser.parse_args()

    with open(args.bootstrap_results_file) as fp:
        bootstrap_results = json.load(fp)
    
    print(bootstrap_results)
    separated_columns = defaultdict(list)
    stringed_colums = defaultdict(list)
    for predictor, results in bootstrap_results.items():
        separated_columns['predictor'].append(predictor)
        stringed_colums['predictor'].append(predictor)
            
        for metric, statistics in results.items():
            low, median, high = statistics['low'], statistics['median'], statistics['high']
            separated_columns[f'{metric}_low'].append(low)
            separated_columns[f'{metric}_median'].append(median)
            separated_columns[f'{metric}_high'].append(high)
            
            stringed_colums[metric].append(f'{median:.02f} (95% CI: {low:.02f} to {high:.02f})')
    sperated_colums_df = pd.DataFrame(data=separated_columns)
    stringed_colums_df = pd.DataFrame(data=stringed_colums)
    
    args.output_directory.mkdir(exist_ok=True, parents=True)
    sperated_colums_df.to_excel(args.output_directory / 'bootstrapped_results_separate_ci.xlsx', index=False)
    stringed_colums_df.to_excel(args.output_directory / 'bootstrapped_results_human_readable_ci.xlsx', index=False)
            
            
            
    
if __name__ == '__main__':
    main()