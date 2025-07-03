import argparse
import json
import multiprocessing
from pathlib import Path
import re
from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import numpy as np
from tqdm import tqdm

def aggregate_annotations(annotations):
    return annotations.values

def get_ci(samples, ci):
    alpha = 1 - ci
    sorted_samples = np.sort(samples)
    low, median, high = np.quantile(sorted_samples, [alpha*0.5, 0.5, 1-alpha*0.5])
    return {'low': low, 'median': median, 'high': high}

def calculate_bootstrap_statistics(targets, bootstrap_samples, ci, discretization_threshold):
    n_subjects, n_bootstraps = bootstrap_samples.shape


    discretized_bootstrap_predictions = bootstrap_samples >= discretization_threshold
    positive_predictions = discretized_bootstrap_predictions == 1.
    negative_predictions = discretized_bootstrap_predictions == 0.
    positive = (targets == 1)[:, np.newaxis]
    negative = (targets == 0)[:, np.newaxis]

    true_positives = (positive_predictions & positive).sum(axis=0)
    true_negatives = (negative_predictions & negative).sum(axis=0)
    false_positives = (positive_predictions & negative).sum(axis=0)
    false_negatives = (negative_predictions & positive).sum(axis=0)

    ## Accuracy
    correct_predictions = discretized_bootstrap_predictions == targets[:, np.newaxis]
    accuracy = correct_predictions.mean(axis=0)
    accuracy_ci = get_ci(accuracy, ci)
    
    ## Positive Predictive Value
    predicted_positive = positive_predictions.sum(axis=0)
    precision = true_positives / predicted_positive
    precision_ci = get_ci(precision, ci)

    ## Negative predictive value
    predicted_positive = positive_predictions.sum(axis=0)
    precision = true_positives / predicted_positive
    precision_ci = get_ci(precision, ci)


    # Stats sensitivty
    positive_indices = targets == 1
    predictions_for_positives = discretized_bootstrap_predictions[positive_indices] 
    sensitivity = predictions_for_positives.mean(axis=0)
    sensitivity_ci = get_ci(sensitivity, ci)

    ## Stats specificity
    negative_indices = targets == 0
    predictions_for_negatives = discretized_bootstrap_predictions[negative_indices] 
    specificity = (1-predictions_for_negatives).mean(axis=0)
    specificity_ci = get_ci(specificity, ci)

    ## F1 measure
    f1 = (2 * precision * sensitivity) / (precision+sensitivity)
    f1_ci = get_ci(f1, ci)

    ## Mathews correlation coefficient/Phi Coefficient
    mcc = (true_positives*true_negatives - false_positives*false_negatives) / np.sqrt((true_negatives+false_positives)*(true_positives+false_negatives)*(true_positives+false_positives)*(true_negatives+false_negatives))
    mcc_ci = get_ci(mcc, ci)
    
    roc_auc_scores = np.array([roc_auc_score(targets, bootstrap_samples[:,i]) for i in range(n_bootstraps)])
    roc_auc_scores_ci = get_ci(roc_auc_scores, ci)
    
    

    return {'accuracy': accuracy_ci, 
            'precision': precision_ci, 
            'sensitivity': sensitivity_ci, 
            'specificity': specificity_ci, 
            'roc_auc': roc_auc_scores_ci,  
            'f1': f1_ci,
            'mcc': mcc_ci}

def make_bootstrap(prediction_file, ci, discretization_threshold, n_bootstraps, random_seed):
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

        bootstrapped_statistics = calculate_bootstrap_statistics(targets.values, bootstrap_means, ci, discretization_threshold)
        return predictions_maker, bootstrapped_statistics

def process_work_package(work_package):
    prediction_file, ci, discretization_threshold, n_bootstraps, random_seed = work_package
    prediction_maker, bootstrapped_statistic = make_bootstrap(prediction_file, ci, discretization_threshold, n_bootstraps, random_seed)
    return (prediction_maker, bootstrapped_statistic)

def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('predictions_directory', help="Directory containing the prediction files", type=Path)
    parser.add_argument('--output-dir', help="Where to write bootstrap results", type=Path, default=Path("analysis") / "bootstraped_statistics")
    parser.add_argument('--bootstrap-runs', help="Number of bootstrap runs", type=int, default=10000)
    parser.add_argument('--confidence-interval', help="Confidence interval as a percentage of bootstrapped distribution", type=float, default=0.95)
    parser.add_argument('--random-seed', help="Random seed for the bootstrap", type=int, default=1729)
    parser.add_argument('--discretization-threshold', help="Threshold on which to determine positive vs. negative predictions", type=float, default=0.5)
    args = parser.parse_args()

    all_predictions = dict()
    files = sorted(args.predictions_directory.glob('**/*.csv'))
    work_packages = [(f, args.confidence_interval, args.discretization_threshold, args.bootstrap_runs, args.random_seed) for f in files]
    
    
    with multiprocessing.Pool() as pool:
        for prediction_maker, bootstrapped_statistics in tqdm(pool.imap_unordered(process_work_package, work_packages), total=len(work_packages)):
            all_predictions[prediction_maker] = bootstrapped_statistics
            
    # for prediction_maker, bootstrapped_statistics in tqdm(map(process_work_package, work_packages), total=len(work_packages)):
    #     all_predictions[prediction_maker] = bootstrapped_statistics
        
    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_file = args.output_dir / 'bootstrapped_statistics.json'
    with open(output_file, 'w') as fp:
        json.dump(all_predictions, fp)
    
    
        
    

if __name__ == '__main__':
    main()