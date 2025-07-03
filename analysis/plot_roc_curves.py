import argparse
import json
import multiprocessing
from pathlib import Path
import re
import itertools
import functools
from collections import defaultdict

from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib


import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

import pandas as pd
import numpy as np

import analysis_config

#sns.set_theme()
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   :42}

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def get_error(x):
    print(x)
    return x

def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('roc_curves_files', help="JSON files containing roc curves", type=Path)
    parser.add_argument('output_directory', help="Where to write bootstrap results", type=Path)
    args = parser.parse_args()


    args.output_directory.mkdir(exist_ok=True, parents=True)
    
    # The data we have are one file per predictor, we instead want to transpose it so the 
    # dict here has the metrics as the root keys, which each points to a list of pd.Series (which will later be converted to a dataframe)

    
    with open(args.roc_curves_files) as fp:
        roc_curves = json.load(fp)
    
    translated_roc_curves = {analysis_config.predictor_translations[predictor]: predictor_roc_curve for predictor, predictor_roc_curve in roc_curves.items()}
    f = plt.figure(figsize=(10,8), dpi=150)
        
    for predictor in analysis_config.predictors:
        predictor_roc_curve = translated_roc_curves[predictor]
        predictor_color = analysis_config.predictor_colors[predictor]
        plt.plot(predictor_roc_curve['fpr'], predictor_roc_curve['tpr'], c=predictor_color, label=predictor, alpha=analysis_config.alpha)
            #plt.tight_layout()
        #ax.figure.legends[0].set_title(translated_subgroup_name)
    
    plt.legend(loc="lower right")
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specificity')
    plt.title("ROC of bootstrapped mean predictions")
    plt.show()
            
            
            
    
if __name__ == '__main__':
    main()