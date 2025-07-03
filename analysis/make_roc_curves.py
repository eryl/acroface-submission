import argparse
import json
import multiprocessing
from pathlib import Path
import re
from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_annotations(annotations):
    return annotations.values


def make_bootstrap(prediction_file, n_bootstraps, random_seed):
        predictions_maker = prediction_file.with_suffix('').name.split('_')[0]
        predictions = pd.read_csv(prediction_file)
        
        # Pandas refuses to accept ndarrays as the result of aggregate on columns which are not of object dtype, so we cast it as such 
        predictions['annotation'] = predictions['annotation'].astype(object)
        
        grouped_annotations = predictions.groupby(by=['subject', 'label']).agg({'annotation': aggregate_annotations}).reset_index()
        
        targets = grouped_annotations['label']
        annotations = list(grouped_annotations['annotation'].values)

        rng = np.random.default_rng(random_seed)

        n_subjects = len(annotations)

        bootstrap_means = np.zeros((n_subjects, n_bootstraps), dtype=np.float64)

        for i, subject_annotations in enumerate(annotations):
            # We will create a index-array which is (n_bootstraps, len(subject_annotations)) 
            # where each row contain the indices for that bootstrap sample
            bootstrap_indices = rng.integers(low=0, high=len(subject_annotations), size=(n_bootstraps, len(subject_annotations)))
            subject_bootstrap = subject_annotations[bootstrap_indices]
            subject_mean_prediction = np.mean(subject_bootstrap, axis=1)
            bootstrap_means[i] = subject_mean_prediction
        per_subject_means = bootstrap_means.mean(axis=1)
        predictor_roc_curve = roc_curve(targets.values, per_subject_means)
        
        return predictions_maker, predictor_roc_curve

def process_work_package(work_package):
    prediction_file, n_bootstraps, random_seed = work_package
    prediction_maker, bootstrapped_statistic = make_bootstrap(prediction_file, n_bootstraps, random_seed)
    return (prediction_maker, bootstrapped_statistic)

def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('predictions_directory', help="Directory containing the prediction files", type=Path)
    parser.add_argument('output_dir', help="Where to write bootstrap results", type=Path)
    parser.add_argument('--bootstrap-runs', help="Number of bootstrap runs", type=int, default=10000)
    parser.add_argument('--random-seed', help="Random seed for the bootstrap", type=int, default=1729)
    args = parser.parse_args()

    all_predictions = dict()
    files = sorted(args.predictions_directory.glob('**/*.csv'))
    work_packages = [(f ,args.bootstrap_runs, args.random_seed) for f in files]
    
    if True:
        with multiprocessing.Pool() as pool:
            for prediction_maker, roc_curve in tqdm(pool.imap_unordered(process_work_package, work_packages), total=len(work_packages)):
                all_predictions[prediction_maker] = roc_curve
    else:            
        for prediction_maker, roc_curve in tqdm(map(process_work_package, work_packages), total=len(work_packages)):
            all_predictions[prediction_maker] = roc_curve
    roc_curves = {}
    for predictor, (fpr, tpr, thresholds) in all_predictions.items():
        roc_curves[predictor] = {'tpr': tpr.tolist(), 'fpr': fpr.tolist(), 'thresholds': thresholds.tolist()}
    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_file = args.output_dir / 'roc_curves.json'
    with open(output_file, 'w') as fp:
        json.dump(roc_curves, fp)
    
    
    
    
        
    

if __name__ == '__main__':
    main()