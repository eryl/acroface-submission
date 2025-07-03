import argparse
import json
import multiprocessing
from pathlib import Path
import re
import itertools
import functools

from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
import matplotlib
import matplotlib.pyplot as plt

import analysis_config

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

def main():
    parser = argparse.ArgumentParser(description="Visualize case predictions as Venn diagrams")
    parser.add_argument('set_file', help="File containing prediction sets", type=Path)
    parser.add_argument('output_dir', help="Where to write bootstrap results", type=Path)
    args = parser.parse_args()

    with open(args.set_file) as fp:
        predictions_info = json.load(fp)
    
    confusion_matrix = predictions_info['confusion_matrix']
    positive_cases = set(predictions_info['positive_cases'])
    negative_cases = set(predictions_info['negative_cases'])

    
    prediction_sets_order = [('true_positives', "True Positive", positive_cases), ('false_negatives', "False Negative", positive_cases), ('false_positives', "False Positive", negative_cases), ('true_negatives', "True Negative", negative_cases)]
    #prediction_sets_order = [('false_negatives', "False Negative", positive_cases), ('false_positives', "False Positive", negative_cases)]
    n_prediction_types = len(prediction_sets_order)
    n_cols = int(np.ceil(np.sqrt(n_prediction_types)))
    n_rows = int(np.floor(n_prediction_types/n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10,10), dpi=150)
    
    annotation_labels = {'human': 'Experts', 'farl': 'FaRL', 'ensemble': "ImageNet\nensemble"}
    
    for ax, (prediction_type, prediction_type_label, reference_cases) in zip(axes.flatten(), prediction_sets_order):
        # This code assumes exactly three predictors, otherwise there will be issues
        named_sets = confusion_matrix[prediction_type]
        sets = []
        set_names = []
        names = []
        set_colors = []
        for name, prediction_set in named_sets.items():
            sets.append(set(prediction_set))
            n_in_set = len(prediction_set)
            annotated_name = f"{annotation_labels[name]}\n{n_in_set}"
            translated_predictor = analysis_config.predictor_translations[name]
            set_color = analysis_config.predictor_colors[translated_predictor]
            set_colors.append(set_color)
            set_names.append(annotated_name)
            names.append(name)
        name_a, name_b, name_c = names
        A, B, C = sets
        Abc = A - (B | C) # Difference between A and the union of B and C
        aBc = B - (A | C)
        ABc = (A & B) - C
        abC = C - (A | B)
        AbC = (A & C) - B
        aBC = (B & C) - A
        ABC = A & B & C
        print(f"{prediction_type} all agree: ", sorted(ABC))
        print(f"{prediction_type} only {name_a}: ", sorted(Abc))
        print(f"{prediction_type} only {name_b}: ", sorted(aBc))
        print(f"{prediction_type} only {name_c}: ", sorted(abC))
        print(f"{prediction_type}, {name_b} and {name_c}: ", sorted(aBC))
        print(f"{prediction_type}, {name_a} and {name_c}: ", sorted(AbC))
        print(f"{prediction_type}, {name_a} and {name_b}: ", sorted(ABc))
        
        
        outside = reference_cases - (A|B|C)
        sets = (len(Abc), len(aBc), len(ABc), len(abC), len(AbC), len(aBC), len(ABC))
        sizes = tuple(np.sqrt(sets) + 2)
        # The order of the argument to the venn diagra, should be (Abc, aBc, ABc, abC, AbC, aBC, ABC), where small letters indicate that set is not in that intersection
        v = venn3(subsets=sets, 
                  set_labels=set_names, 
                  ax=ax, 
                  set_colors=set_colors, 
                  alpha=0.7, #analysis_config.alpha, 
                  layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=sizes), )
        for p in v.patches:
            p.set_linestyle('solid')
            p.set_edgecolor('white')
        #v.get_patch_by_id('100').set_color('green')
        #venn3(sets, names,ax=ax )
              #layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1,1,1,1,1)), ax=ax)
        
        #ax.set_title(f"{prediction_type_label}\nTotal number of cases: {len(reference_cases)}")
        ax.text(0.5, 1.08, f"{prediction_type_label}", fontsize=20, ha='center', transform=ax.transAxes)
        ax.text(0.5, 1.01, f"Total number of cases: {len(reference_cases)}", fontsize=14, ha='center', transform=ax.transAxes)
    
    plt.subplots_adjust(top=0.94,
                        bottom=0.0,
                        left=0.1,
                        right=0.94,
                        hspace=0.1,
                        wspace=0.5)
    plt.show()
    
        
    
if __name__ == '__main__':
    main()