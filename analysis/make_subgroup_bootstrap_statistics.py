import argparse
from collections import defaultdict
import functools
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
    n_positive = positive.sum()
    n_negative = negative.sum()
    
    true_positives = (positive_predictions & positive).sum(axis=0)
    true_negatives = (negative_predictions & negative).sum(axis=0)
    false_positives = (positive_predictions & negative).sum(axis=0)
    false_negatives = (negative_predictions & positive).sum(axis=0)

    true_positive_rate = true_positives / n_positive
    true_negative_rate = true_negatives / n_negative
    
    
    ## Accuracy
    correct_predictions = discretized_bootstrap_predictions == targets[:, np.newaxis]
    accuracy = correct_predictions.mean(axis=0)
    accuracy_ci = get_ci(accuracy, ci)
    
    # Balanced Accuracy
    balanced_accuracy = (true_positive_rate + true_negative_rate)/2
    balanced_accuracy_ci = get_ci(balanced_accuracy, ci)
    
    ## Precision/Positive Predictive Value
    predicted_positive = positive_predictions.sum(axis=0)
    precision = true_positives / predicted_positive
    precision_ci = get_ci(precision, ci)
    
    ## Negative Predictive Value
    predicted_negative = negative_predictions.sum(axis=0)
    npv = false_negatives / predicted_negative
    npv_ci = get_ci(npv, ci)

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
    
    with multiprocessing.Pool() as pool:
        roc_auc_partial = functools.partial(roc_auc_score, targets)
        results = []
        for result in tqdm(pool.imap_unordered(roc_auc_partial, bootstrap_samples.T), total=n_bootstraps):
            results.append(result)
        roc_auc_scores_ci = get_ci(np.array(results), ci)
    roc_auc_scores = np.array([roc_auc_score(targets, bootstrap_samples[:,i]) for i in range(n_bootstraps)])
    
    return {
        'accuracy': accuracy_ci, 
        'precision': precision_ci, 
        'sensitivity': sensitivity_ci, 
        'specificity': specificity_ci, 
        'roc_auc': roc_auc_scores_ci,  
        'f1': f1_ci,
        'mcc': mcc_ci,
        'ba': balanced_accuracy_ci,
        'npv': npv_ci,
        }

def make_bootstrap(prediction_file, ci, discretization_threshold, n_bootstraps, random_seed, subgroup_columns, metadata):
        predictions_maker = prediction_file.with_suffix('').name.split('_')[0]
        predictions = pd.read_csv(prediction_file)
        predictions = predictions.set_index('subject')
        # Pandas refuses to accept ndarrays as the result of aggregate on columns which are not of object dtype, so we cast it as such 
        predictions['annotation'] = predictions['annotation'].astype(object)
        
        grouped_annotations = predictions.groupby(by=['subject', 'label']).agg({'annotation': aggregate_annotations}).reset_index(1)  # We reset the index at the second level, "label", so it becomes a column again
        
        targets = grouped_annotations['label']
        annotations = list(grouped_annotations['annotation'].values)

        rng = np.random.default_rng(random_seed)
        selected_metadata = metadata.loc[targets.index]
        
        n_subjects = len(annotations)

        bootstrap_means = np.zeros((n_subjects, n_bootstraps), dtype=np.float64)

        for i, subject_annotations in enumerate(annotations):
            # We will create a index-array which is (n_bootstraps, len(subject_annotations)) 
            # where each row contain the indices for that bootstrap sample
            bootstrap_indices = rng.integers(low=0, high=len(subject_annotations), size=(n_bootstraps, len(subject_annotations)))
            subject_bootstrap = subject_annotations[bootstrap_indices]
            subject_mean_prediction = np.mean(subject_bootstrap, axis=1)
            bootstrap_means[i] = subject_mean_prediction
        
        
        subgroup_results = defaultdict(dict)
        
        # Make one run without any subgroups
        subgroup_results[None][None] = calculate_bootstrap_statistics(targets.values, bootstrap_means, ci, discretization_threshold)
        
        # Now ad the subgroups
        for subgroup_column in tqdm(subgroup_columns, desc='subgroup'):
            # Aggregate the values for the group but filter out n/a
            subgroup = selected_metadata[subgroup_column]
            subgroup_values = set(subgroup[~subgroup.isna()])
            for subgroup_value in tqdm(subgroup_values, desc='subgroup_values'):
                subgroup_indices = subgroup == subgroup_value
                subgroup_targets = targets[subgroup_indices]
                subgroup_bootstrap = bootstrap_means[subgroup_indices]
                subgroup_results[subgroup_column][subgroup_value] = calculate_bootstrap_statistics(subgroup_targets.values, subgroup_bootstrap, ci, discretization_threshold)
        
                
        return predictions_maker, subgroup_results

def process_work_package(work_package):
    prediction_file, ci, discretization_threshold, n_bootstraps, random_seed, subgroup_columns, metadata, output_dir = work_package
    prediction_maker, bootstrapped_statistic = make_bootstrap(prediction_file, ci, discretization_threshold, n_bootstraps, random_seed, subgroup_columns, metadata)
    output_file = output_dir / f'subgrouped_bootstrapped_statistics_{prediction_maker}.json'
    with open(output_file, 'w') as fp:
        bootstrap_info = {prediction_maker: bootstrapped_statistic}
        json.dump(bootstrap_info, fp)
    
    return (prediction_maker, bootstrapped_statistic)

def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('predictions_directory', help="Directory containing the prediction files", type=Path)
    parser.add_argument('metadata_file', help="Directory containing the prediction files", type=Path)
    parser.add_argument('output_dir', help="Where to write bootstrap results", type=Path)
    parser.add_argument('--bootstrap-runs', help="Number of bootstrap runs", type=int, default=10000)
    parser.add_argument('--confidence-interval', help="Confidence interval as a percentage of bootstrapped distribution", type=float, default=0.95)
    parser.add_argument('--random-seed', help="Random seed for the bootstrap", type=int, default=1729)
    parser.add_argument('--discretization-threshold', help="Threshold on which to determine positive vs. negativ predictions", type=float, default=0.5)
    parser.add_argument('--subgroup-columns', help="The subgroups to look at", nargs='+', default=('gender', 'age groups 2', 'age groups 3', 'age groups 4', "bioactive pair"))
    args = parser.parse_args()

    all_predictions = dict()
    files = sorted(args.predictions_directory.glob('**/*.csv'))
    metadata = pd.read_csv(args.metadata_file)
    metadata_reindexed = metadata.set_index('subject')
    args.output_dir.mkdir(exist_ok=True, parents=True)
    work_packages = [(f, args.confidence_interval, args.discretization_threshold, args.bootstrap_runs, args.random_seed, args.subgroup_columns, metadata_reindexed, args.output_dir) for f in files]
    
    if False:
        with multiprocessing.Pool() as pool:
            for prediction_maker, bootstrapped_statistics in tqdm(pool.imap_unordered(process_work_package, work_packages), total=len(work_packages)):
                pass
    else:            
        for prediction_maker, subgroup_statistics in tqdm(map(process_work_package, work_packages), desc='predictor', total=len(work_packages)):
            pass
            

    

if __name__ == '__main__':
    main()