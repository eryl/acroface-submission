import argparse
import json
from pathlib import Path
import re
from collections import defaultdict

from sklearn.metrics import roc_auc_score, roc_curve


import pandas as pd
import numpy as np

import analysis_config


def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('bootstrap_results_dir', help="Directory containing the prediction files", type=Path)
    parser.add_argument('output_directory', help="Where to write bootstrap results", type=Path)
    
    args = parser.parse_args()

    separated_columns = defaultdict(lambda: defaultdict(list))
    human_readable_columns = defaultdict(lambda: defaultdict(list))
    args.output_directory.mkdir(exist_ok=True, parents=True)
    
    for f in args.bootstrap_results_dir.glob('**/*.json'):
        m = re.match(r".*_([\w\-]+)\.json", f.name)
        if m is not None:
            predictor, = m.groups()
        else:
            raise ValueError(f"Can't match {f.name}")
        
        with open(f) as fp:
            bootstrap_info = json.load(fp)
        
        for predictor, bootstrap_results in bootstrap_info.items():
            predictor_translated = analysis_config.predictor_translations[predictor]
            if predictor_translated not in analysis_config.predictors:
                print(f"Skipping predictor {predictor_translated}")
                continue
            for subgroup_name, subgroup_value_results in bootstrap_results.items():
                subgroup_separated_columns = separated_columns[subgroup_name]
                subgroup_human_readable_columns = human_readable_columns[subgroup_name]
                
                
                for subgroup_value, results in subgroup_value_results.items():
                    subgroup_separated_columns['predictor'].append(predictor)
                    subgroup_human_readable_columns['predictor'].append(predictor)  
                    subgroup_separated_columns[subgroup_name].append(subgroup_value)
                    subgroup_human_readable_columns[subgroup_name].append(subgroup_value)
                    for metric, statistics in results.items():
                        # metric_translated = analysis_config.metrics_translation.get(metric, None)
                        # if metric_translated not in analysis_config.metrics:
                        #     print(f"Skipping metric {metric_translated}")
                        #     continue
                        low, median, high = statistics['low'], statistics['median'], statistics['high']
                        subgroup_separated_columns[f'{metric}_low'].append(low)
                        subgroup_separated_columns[f'{metric}_median'].append(median)
                        subgroup_separated_columns[f'{metric}_high'].append(high)
                        
                        subgroup_human_readable_columns[metric].append(f'{median:.02f} (95% CI: {low:.02f} to {high:.02f})')
    for subgroup_name, subgroup_separated_columns in separated_columns.items():
        sperated_colums_df = pd.DataFrame(data=subgroup_separated_columns)
        if subgroup_name == 'null':
            output_file = args.output_directory / f'results_seperate_ci.xlsx'
            sperated_colums_df = sperated_colums_df.drop(columns=[subgroup_name])
        else:
            output_file = args.output_directory / f'subgroup_results_{subgroup_name}_seperate_ci.xlsx'
        
        sperated_colums_df.to_excel(output_file, index=False)
    
    for subgroup_name, subgroup_human_readable_columns in human_readable_columns.items():
        sperated_colums_df = pd.DataFrame(data=subgroup_human_readable_columns)
        if subgroup_name == 'null':
            output_file = args.output_directory / f'results_human_readable.xlsx'
            sperated_colums_df = sperated_colums_df.drop(columns=[subgroup_name])
        else:
            output_file = args.output_directory / f'subgroup_results_{subgroup_name}_human_readable.xlsx'
        sperated_colums_df.to_excel(output_file, index=False)
        
        
            
            
            
    
if __name__ == '__main__':
    main()