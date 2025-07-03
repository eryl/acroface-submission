import argparse
import json
import multiprocessing
from pathlib import Path
import re
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_performance_statistics(grouped_predictions):
    print(grouped_predictions)

    predictions = grouped_predictions['annotation'].values
    targets = grouped_predictions['label'].values
    positive_predictions = predictions == 1.
    negative_predictions = predictions == 0.
    positive = (targets == 1)
    negative = (targets == 0)
    n_positive = positive.sum()
    n_negative = negative.sum()

    true_positives = (positive_predictions & positive).sum()
    true_negatives = (negative_predictions & negative).sum()
    false_positives = (positive_predictions & negative).sum()
    false_negatives = (negative_predictions & positive).sum()
    
    true_positive_rate = true_positives / n_positive
    true_negative_rate = true_negatives / n_negative

    ## Accuracy
    correct_predictions = predictions == targets
    accuracy = correct_predictions.mean()
    accuracy_sk = accuracy_score(targets, predictions)
    
    # Balanced Accuracy
    balanced_accuracy = (true_positive_rate + true_negative_rate)/2
        
    ## Negative Predictive Value
    predicted_negative = negative_predictions.sum(axis=0)
    npv = true_negatives / predicted_negative
    
    ## Precision
    predicted_positive = positive_predictions.sum()
    precision = true_positives / predicted_positive
    precision_sk = precision_score(targets, predictions)

    # Stats sensitivty
    positive_indices = targets == 1
    predictions_for_positives = predictions[positive_indices] 
    sensitivity = predictions_for_positives.mean()
    sensitivity_sk = recall_score(targets, predictions)

    ## Stats specificity
    negative_indices = targets == 0
    predictions_for_negatives = predictions[negative_indices] 
    specificity = (1-predictions_for_negatives).mean()
    
    ## F1 measure
    f1 = (2 * precision * sensitivity) / (precision+sensitivity)
    f1_sk = f1_score(targets, predictions)
    
    ## Mathews Correlation Coefficient
    mcc = (true_positives*true_negatives - false_positives*false_negatives) / np.sqrt((true_negatives+false_positives)*(true_positives+false_negatives)*(true_positives+false_positives)*(true_negatives+false_negatives))
    mcc_sk = matthews_corrcoef(targets, predictions)
    
    roc_auc_performance = roc_auc_score(targets, predictions)
    performance_series = pd.Series({'accuracy': accuracy, 
                                   #'accuracy_sk': [accuracy_sk], 
                                   'ppv': precision, 
                                   #'precision_sk': [precision_sk], 
                                   'sensitivity': sensitivity, 
                                   #'sensitivity_sk': [sensitivity_sk], 
                                   'specificity': specificity, 
                                   'roc_auc': roc_auc_performance,  
                                   'f1': f1, 
                                   #'f1_sk': [f1_sk], 
                                   'mcc': mcc,
                                   #'mcc_sk': [mcc_sk]
                                   'ba': balanced_accuracy,
                                   'npv': npv,
                                   })

    return performance_series

def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('predictions_directory', help="Directory containing the prediction files", type=Path)
    parser.add_argument('output_dir', help="Where to write bootstrap results", type=Path)
    args = parser.parse_args()

    all_predictions = dict()
    files = sorted(args.predictions_directory.glob('**/*.csv'))

    args.output_dir.mkdir(exist_ok=True, parents=True)
    for f in files:
        df = pd.read_csv(f)
        performances = df.groupby(by='annotator').apply(calculate_performance_statistics).reset_index()
        output_file = args.output_dir / f"{f.with_suffix('').name}_individual_performance.xlsx"
        performances.to_excel(output_file, index=False)
        #print(df)
            

    

if __name__ == '__main__':
    main()