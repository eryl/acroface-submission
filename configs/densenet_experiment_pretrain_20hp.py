from acroface.experiment.core import ExperimentConfig, HPOConfig
from acroface.dataset.example import ExampleDataset
from acroface.dataset.resample import DefaultStrategy, KFoldStrategy, GroupKFoldStrategy, GroupSubsampleStrategy
from acroface.evaluation.constants import ROC_AUC

top_resample_strategy = GroupKFoldStrategy(k=10)
nested_resample_strategy = GroupKFoldStrategy(k=10, n_runs=1)

hpo_resample_root_strategy = GroupSubsampleStrategy(n_runs=1, test_size=0.1)
hpo_resample_nested_strategy = GroupSubsampleStrategy(n_runs=1, test_size=0.1)
hpo_config = HPOConfig(hp_iterations=20, 
                       hp_optimization_metric=ROC_AUC,
                       hp_direction='maximize', 
                       root_resample_strategy=hpo_resample_root_strategy, 
                       nested_resample_strategy=hpo_resample_nested_strategy)


dataset_config_path = "configs/datasets/acroface/acroface_ml.py"
model_config_path = "configs/models/densenet_classifier_pretrain.py"

experiment_config = ExperimentConfig(name='densenet_experiment_pretrain_20hp', 
                                     dataset_config_path=dataset_config_path,
                                     data_split_path='configs/dataset_splits/base_splits.json',
                                     model_config_path=model_config_path,
                                     resample_strategy=top_resample_strategy,
                                     nested_resample_strategy=nested_resample_strategy,
                                     hpo_config=hpo_config,
                                     multiprocessing_envs=[#{'CUDA_VISIBLE_DEVICES': '0'},
                                                           #{'CUDA_VISIBLE_DEVICES': '1'},
                                                           #{'CUDA_VISIBLE_DEVICES': '2'},
                                                           #{'CUDA_VISIBLE_DEVICES': '3'},
                                                           #{'CUDA_VISIBLE_DEVICES': '4'},
                                                           #{'CUDA_VISIBLE_DEVICES': '5'},
                                                           #{'CUDA_VISIBLE_DEVICES': '6'},
                                                           #{'CUDA_VISIBLE_DEVICES': '7'},
                                                           ]
)
