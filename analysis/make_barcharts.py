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
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

import pandas as pd
import numpy as np

sns.set_theme()

def get_error(x):
    print(x)
    return x

def main():
    parser = argparse.ArgumentParser(description="Perform bootstrap analysis based on the prediction files in the given directory")
    parser.add_argument('bootstrap_results_dir', help="Directory containing the prediction files", type=Path)
    parser.add_argument('output_directory', help="Where to write bootstrap results", type=Path)
    args = parser.parse_args()


    args.output_directory.mkdir(exist_ok=True, parents=True)
    
    # The data we have are one file per predictor, we instead want to transpose it so the 
    # dict here has the metrics as the root keys, which each points to a list of pd.Series (which will later be converted to a dataframe)
    transposed_results = defaultdict(list)

    predictor_translations = {
                              'inceptionv3': "InceptionV3", 
                              'resnet': "ResNet-52", 
                              'human': "Endocrinologists", 
                              'ensemble': "ImageNet Ensemble",
                              'ensemble-and-farl': "ImageNet Ensemble+FaRL",
                              'svm-densnet': "SVM on DensNet", 
                              'knn-densenet': "kNN on DenseNet", 
                              'rf-densenet': "Random forest on DenseNet", 
                              'densenet': "DenseNet-121", 
                              'farl': "FaRL", 
                              'svm-farl': "SVN on FaRL representations", 
                              'knn-farl': "kNN on FaRL representations", 
                              'rf-farl': "Random forest on FaRL representations"
                              }
    
    subgroup_value_translations = {'gender': {'name': 'Gender',
                                               'translation': {'0': 'Female',
                                                               '1': 'Male'},
                                                       'order': ['Female', 'Male']},
                                   'bioactive pair': {'name': 'Biochemically\nActive',
                                                       'translation': {'0': 'No',
                                                               '1': 'Yes'},
                                                       'order': ['Yes', 'No']
                                                       
                                                },
                                   'age groups 2' : {'name': 'Age', # Groups, (23-58, 59-86)
                                                     'order': ['23-58', '59-86']},
                                   'age groups 3' : {'name': 'Age', # : (23-51, 52-64,  65-86)
                                                       'order': ['23-51', '52-64',  '65-86']},
                                   'age groups 4' : {'name': 'Age', # : (23-46, 47-58, 59-67, 68-86)
                                                       'order': ['23-46', '47-58', '59-67', '68-86']},
                                   'null': {'name': 'null',
                                            'order': ['null']},
                                  }
    
    metrics_translation = {'accuracy': "Accuracy", 
               'f1': "F1-score", 
               'precision': "Positive Predictive Value", 
               'roc_auc': "ROC AUC", 
               'mcc': "MCC", 
               'specificity': "Specificity", 
               'sensitivity': "Sensitivity",
               'ba': 'Balanced Accuracy'}
    
    metrics_intervals = {"Accuracy": (0.5, 1), 
               "F1-score": (0.5, 1), 
               "Precision": (0.5, 1), 
               "ROC AUC": (0.5, 1), 
               "MCC": (-1, 1), 
               "Specificity": (0.5, 1), 
               "Sensitivity": (0.5, 1),
               'Balanced Accuracy': (0.5, 1)}
    
    predictors = ["ResNet-52",
                  "InceptionV3", 
                  "DenseNet-121", 
                  "ImageNet Ensemble",  
                  "FaRL",
                  "ImageNet Ensemble+FaRL",
                  # Ensemble + FaRL
                  "Endocrinologists", 
                  ]
    
    subgroups = [
                 'gender',
                 'bioactive pair',
                 'age groups 3',
                 'null'
                ]
    
    metrics = ["ROC AUC", "Sensitivity", "Specificity",  "Balanced Accuracy"] # 
    subgroup_value_summary = defaultdict(set)
    predictor_set = set()
    for f in args.bootstrap_results_dir.glob('**/*.json'):
        with open(f) as fp:
            bootstrap_info = json.load(fp)
        
     
        for predictor, bootstrap_results in bootstrap_info.items():
            predictor_set.add(predictor)
            for subgroup_name, subgroup_value_results in bootstrap_results.items():
                for subgroup_value, results in subgroup_value_results.items():
                    for metric, statistics in results.items():
                        translated_metric = metrics_translation.get(metric, metric)
                        translated_predictor = predictor_translations[predictor]
                        if translated_metric not in metrics or translated_predictor not in predictors:
                            continue
                        low, median, high = statistics['low'], statistics['median'], statistics['high']
                        translated_subgroup_name = subgroup_name
                        translated_subgroup_value = subgroup_value
                        if subgroup_name in subgroup_value_translations:
                            translated_subgroup_name = subgroup_value_translations[subgroup_name]['name']
                            if 'translation' in subgroup_value_translations[subgroup_name] and subgroup_value in subgroup_value_translations[subgroup_name]['translation']:
                                translated_subgroup_value = subgroup_value_translations[subgroup_name]['translation'][subgroup_value]
                        subgroup_value_summary[translated_subgroup_name].add(translated_subgroup_value)
                        subgroup_values_order = subgroup_value_translations[subgroup_name]['order']
                        s = pd.Series({'low': low, 'high': high, 'median': median, 'predictor': translated_predictor, 'subgroup_name': subgroup_name, 'subgroup_value': translated_subgroup_value, 'predictor_order': predictors.index(translated_predictor), 'subgroup_order': subgroup_values_order.index(translated_subgroup_value)})
                        transposed_results[translated_metric].append(s)
                        
    plt.set_cmap('viridis')
    for metric_name in metrics:
        (y_min, y_max) = metrics_intervals[metric_name]
        metric_series = transposed_results[metric_name]
        metric_df = pd.DataFrame(metric_series)
        #subgroups = sorted(set(metric_df['subgroup_name']))
        
        for subgroup in subgroups:
            subgroup_df = metric_df[(metric_df['subgroup_name'] == subgroup) & metric_df['predictor'].isin(predictors)].rename(columns={'subgroup_value': subgroup})
            subgroup_df = subgroup_df.sort_values(by=['predictor_order', 'subgroup_order'])
            
            translated_subgroup_name = subgroup
            if subgroup in subgroup_value_translations:
                translated_subgroup_name = subgroup_value_translations[subgroup]['name']
            
            subgroup_values = set(subgroup_df[subgroup])
            

            #subgroup_df = subgroup_df.rename(columns={'subgroup_name': subgroup})
            output_file = args.output_directory / f'subgroup_results_{metric_name}_{subgroup}.png'
            
            #ax = sns.barplot(subgroup_df, x='predictor', y='median', hue=subgroup, palette='viridis', alpha=.8, errorbar=get_error)
            
            # #Establish order of x-values and hues; retreive number of hues
            # order = predictors; hue_order=subgroup_value_translations[subgroup]['order']; n_hues = len(hue_order)
            # # Get the bar width of the plot
            # bar_width = ax.patches[0].get_width()
            # #Calculate offsets for number of hues provided
            # offset = np.linspace(-n_hues / 2, n_hues / 2, n_hues)*bar_width*n_hues/(n_hues+1); # Scale offset by number of hues, (dividing by n_hues+1 accounts for the half bar widths to the left and right of the first and last error bars.
            # #Create dictionary to map x values and hues to specific x-positions and offsets
            # x_dict = dict((x_val,x_pos) for x_pos,x_val in list(enumerate(order)))
            # hue_dict = dict((hue_pos,hue_val) for hue_val,hue_pos in list(zip(offset,hue_order)))
            # #Map the x-position and offset of each record in the dataset
            # x_values = np.array([x_dict[x] for x in subgroup_df['predictor']])
            # hue_values = np.array([hue_dict[x] for x in subgroup_df[subgroup]])
            # #Overlay the error bars onto plot
            # ax.errorbar(x = x_values+hue_values, y = subgroup_df['median'], yerr=0.05, fmt='none', c= 'black', capsize = 2)

            
            #plt.tight_layout()
            #ax.figure.legends[0].set_title(translated_subgroup_name)
            #plt.show()
            #f, ax = plt.subplots(1,1, figsize=(12,12))
            f = plt.figure(figsize=(8,6))
            newline_stripped_subname = translated_subgroup_name.replace('\n', ' ')
            title = f"{metric_name} ({newline_stripped_subname})"
            
            if subgroup == 'null':
                subgroup = None
                title = f"{metric_name}"
                output_file = args.output_directory / f'results_{metric_name}.png'
                
            p = (
                so.Plot(data=subgroup_df, y='predictor', x='median', color=subgroup, xmin="low", xmax="high")
                .on(f)
                
                .add(so.Bar(), so.Dodge())
                .add(so.Range(), so.Dodge())
                
                    .scale(color="viridis")
                .label(
                    title=title,
                    legend=translated_subgroup_name,
                )
                .limit(x=(y_min, y_max))
                .layout(size=(12,12), engine='constrained')
                .plot()

            )
            if f.legends:
                legend = f.legends[0]
                legend.set_loc('upper right')
                legend.set_alignment('center')
                legend.set_in_layout(True)
                legend.set_bbox_to_anchor((0.97, 0.96))
                legend.set_title(translated_subgroup_name)
                
            #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            #plt.title(f"{metric_name} ({translated_subgroup_name})")
            #plt.ylim(y_min, y_max)
            #plt.xticks(rotation=-60)
            #ax.get_legend().set_title(translated_subgroup_name)

            #ax.figure.legends[0].set_title(translated_subgroup_name)
            f.savefig(output_file, dpi=150)
            plt.close(f)
            #plt.show()
                
                #p.show()

        
        
            
            
            
    
if __name__ == '__main__':
    main()