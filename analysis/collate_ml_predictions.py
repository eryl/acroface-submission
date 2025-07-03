import argparse
from collections import defaultdict, Counter
import multiprocessing.pool
from pathlib import Path
import re
import multiprocessing

from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_subject(image_path):
    image_path = Path(image_path.strip())
    m = re.match(r"(\w+-[AK])-.*", image_path.name)
    if m is not None:
        subject, = m.groups()
        return subject
    else:
        raise RuntimeError(f"Could not match {image_path.name}")

def get_majority_item(annotations):
    annotation_counts = Counter(annotations)
    [(majority_annotation, count)] = annotation_counts.most_common(1)
    return majority_annotation

def logistic(x):
    return 1/(1 + np.exp(-x))

def calculate_test_predictions(table_dir):
    dev_predictions = pd.read_csv(table_dir / 'dev_predictions.csv')
    test_predictions = pd.read_csv(table_dir / 'test_predictions.csv')
    
    if 'logits' in dev_predictions:
        dev_predictions['p'] = dev_predictions['logits'].map(logistic)
        test_predictions['p'] = test_predictions['logits'].map(logistic)
    
    # We find a discretization threshold using the Youden J statistic. To do this we first calcualte the false positive rate and false negative rate using roc_curve
    fpr, tpr, thresholds = roc_curve(dev_predictions['y'], dev_predictions['p'])
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    discretized_test_predictions = test_predictions['p'] >= best_thresh
    test_predictions['annotation'] = discretized_test_predictions.astype(int)
    test_predictions['subject'] = test_predictions['image_path'].map(get_subject)
    grouped_predictions = test_predictions.groupby(by=['subject']).aggregate({'annotation': get_majority_item, 'y': get_majority_item})
    return grouped_predictions

def process_work_package(work_package):
    experiment_name, timestamp, outer_fold, inner_fold, tables_directory = work_package
    grouped_predictions = calculate_test_predictions(tables_directory)
    renamed_df = grouped_predictions.rename(columns={'y': 'label'})
    renamed_df['annotator'] = f"{experiment_name}"
    renamed_df = renamed_df.reset_index()
    return (experiment_name, timestamp, outer_fold, inner_fold, renamed_df)
    

def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('experiments_directory', help="Directory containing the prediction files", type=Path)
    parser.add_argument('--output-path', help="Where should the collated annotations be written?", type=Path, default=Path("analysis") / 'annotations' / 'deep_learning_annotations')
    parser.add_argument('--multiprocessing', help="If set, the processing will be done in parallell", action='store_true')
    
    args = parser.parse_args()

    tables_dirs = sorted(args.experiments_directory.glob('**/evaluation_resample/[0-9]/tables'))
    work_packages = []
    for d in tables_dirs:
        # This matches out all the interesting parts of the path to deduce model and experiment run
        m = re.match(r"\w+\/([\w_]+)\/([\w\-\.]+)\/children\/(\d+)\/evaluation_resample\/(\d+)\/tables", str(d))
        if m is not None:
            experiment_name, experiment_timestamp, outer_fold, inner_fold = m.groups()
            work_package = (experiment_name, experiment_timestamp, outer_fold, inner_fold, d)
            work_packages.append(work_package)
    
    collated_annotations = defaultdict(list)
    
    if args.multiprocessing:
        with multiprocessing.Pool() as pool:
            for (experiment_name, experiment_timestamp, outer_fold, inner_fold, annotations) in tqdm(pool.imap_unordered(process_work_package, work_packages), total=len(work_packages)):
                collated_annotations[experiment_name].append(annotations)
                
    else:
        for (experiment_name, experiment_timestamp, outer_fold, inner_fold, annotations) in tqdm(map(process_work_package, work_packages), total=len(work_packages)):
            collated_annotations[experiment_name].append(annotations)
    
    args.output_path.mkdir(exist_ok=True, parents=True)
    for experiment_name, all_annotations in collated_annotations.items():
        concatenated_anotations = pd.concat(all_annotations)
        output_path = args.output_path / f"{experiment_name}_annotations.csv"
        concatenated_anotations.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    main()