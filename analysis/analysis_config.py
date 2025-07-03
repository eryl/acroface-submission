from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


predictor_translations = {
                            'inceptionv3': "InceptionV3", 
                            'resnet': "ResNet-52", 
                            'human': "Human experts", 
                            'ensemble': "ImageNet Ensemble", 
                            'svm-densnet': "SVM on DensNet", 
                            'knn-densenet': "kNN on DenseNet", 
                            'rf-densenet': "Random forest on DenseNet", 
                            'densenet': "DenseNet-121", 
                            'farl': "FaRL", 
                            'svm-farl': "SVN on FaRL representations", 
                            'knn-farl': "kNN on FaRL representations", 
                            'rf-farl': "Random forest on FaRL representations",
                            'ensemble-and-farl': "ImageNet Ensemble+FaRL",
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
            'precision': "Precision", 
            'roc_auc': "ROC AUC", 
            'mcc': "MCC", 
            'specificity': "Specificity", 
            'ba': "Balanced Accuracy", 
            'sensitivity': "Sensitivity"}

metrics_intervals = {"Accuracy": (0.5, 1), 
            "F1-score": (0.5, 1), 
            "Precision": (0.5, 1), 
            "ROC AUC": (0.5, 1), 
            "MCC": (-1, 1), 
            "Specificity": (0.5, 1), 
            "Sensitivity": (0.5, 1),
            "Balanced Accuracy": (0.5, 1)}

metrics = {"Accuracy", 
            "F1-score", 
            "Precision", 
            "ROC AUC", 
            "MCC", 
            "Specificity", 
            "Sensitivity",
            "Balanced Accuracy"}

predictors = [#"ResNet-52",
            #"InceptionV3", 
            #"DenseNet-121", 
            "FaRL", 
            "Human experts", 
            "ImageNet Ensemble+FaRL",
            "ImageNet Ensemble",  
                ]

predictor_colors_red_orange_blue_green = {#"ResNet-52",
            #"InceptionV3", 
            #"DenseNet-121", 
            "FaRL": "#476FAE", 
            "Human experts": "#DC7E49", 
            "ImageNet Ensemble+FaRL": "#4BA45F",
            "ImageNet Ensemble": "#C44F53"}

plasma = plt.get_cmap("plasma")
predictor_colors_plasma = {#"ResNet-52",
            #"InceptionV3", 
            #"DenseNet-121", 
            "FaRL": plasma(0.5), 
            "Human experts": plasma(0), 
            "ImageNet Ensemble+FaRL": plasma(0.7),
            "ImageNet Ensemble": plasma(0.95)}

#predictor_colors = {predictor: cmap(x) for predictor, x in zip(predictors, np.linspace(0, 0.95, len(predictors), endpoint=True))}

tab10 = plt.get_cmap("tab10")
predictor_colors_tab10 = {#"ResNet-52",
            #"InceptionV3", 
            #"DenseNet-121", 
            "FaRL": tab10(0), 
            "Human experts": tab10(0.1), 
            "ImageNet Ensemble+FaRL": tab10(0.4),
            "ImageNet Ensemble": tab10(0.2)}

predictor_colors = predictor_colors_tab10


alpha = .9


# Replace the alpha value
#predictor_colors = {predictors: x[:3] + (alpha,) for predictors, x in predictor_colors.items()}            

subgroup_value_summary = defaultdict(set)