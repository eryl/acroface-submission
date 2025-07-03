from collections import defaultdict
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import queue
from typing import Literal, Optional, List, Dict
from functools import partial
import random
import math
import statistics
#import multiprocessing.dummy as multiprocessing
import multiprocessing


#from hoopo.util import load_object
#from hoopo.optuna import suggest_values, best_values
import optuna
from tqdm import tqdm

from acroface.experiment.tracker import ExperimentTracker, find_experiments
from acroface.experiment.hyperparameters import suggest_values, best_values, get_sampler, create_study, set_values, get_hp_params
from acroface.dataset.core import load_dataset, DatasetConfig, AbstractDataset
from acroface.dataset.resample import ResampleStrategy
from acroface.train import run_training
from acroface.models.wrapper import ModelWrapperConfig
from acroface.util import load_object,  timestamp


@dataclass 
class HPOConfig:
    hp_iterations: int
    hp_optimization_metric: str
    hp_direction: Literal['maximize', 'minimize']
    root_resample_strategy: Optional[ResampleStrategy] = None
    nested_resample_strategy: Optional[ResampleStrategy] = None
    aggregation_method: Literal['mean', 'max', 'min', 'median', 'stdev'] = 'mean'
    backend: Literal['optuna'] = 'optuna'
    sampler: Literal['tpe', 'random'] = 'tpe'
    save_model: bool = False


@dataclass
class ExperimentConfig:
    name: str
    dataset_config_path: Path
    model_config_path: Path
    hpo_config: HPOConfig
    resample_strategy: ResampleStrategy
    nested_resample_strategy: ResampleStrategy = None
    data_split_path: Path = None
    random_seed: Optional[int] = None
    multiprocessing_envs: List[Dict[str, str]] = field(default_factory=list)


def resample_worker(work_queue, results_queue, environment):
    print("Worker started with environment", environment)
    for var, value in environment.items():
        os.environ[var] = value
    print("Size of the work queue:", work_queue.qsize())
    while not work_queue.empty():
        try:
            work_package_path = work_queue.get_nowait()
            experiment_on_resample(work_package_path)
        except queue.Empty:
            break


def setup_work_package(*, 
                        experiment_tracker: ExperimentTracker,
                        experiment_config: ExperimentConfig,
                        dataset: AbstractDataset,
                        dataset_splits,
                        random_seed=None,
                    ):
    """Prepare the work package by setting up the files in the target directory.
    """
    if random_seed is None:
        random_seed = random.randint(2**31)
    work_package = dict(experiment_config=experiment_config,
                        dataset=dataset,
                        dataset_splits=dataset_splits,
                        random_seed=random_seed)
    experiment_tracker.log_artifact("work_package", work_package)
    experiment_tracker.set_flag("work_is_done", False)
    return experiment_tracker.output_dir


def experiment_on_resample(experiment_tracker_dir: Path):
    experiment_tracker = ExperimentTracker(experiment_tracker_dir)
    work_package = experiment_tracker.load_artifact('work_package')
    
    random_seed = work_package['random_seed']
    experiment_config = work_package['experiment_config']
    dataset = work_package['dataset']
    dataset_split = work_package['dataset_splits']
    rerun_status = work_package.get('rerun_status', None)
    
    rng = random.Random(random_seed)
    hpo_rng = random.Random(rng.randint(0, 2^31))

    best_config = hpo_search(experiment_config, dataset, dataset_split['hpo_resamples'], experiment_tracker, hpo_rng, rerun_status=='full')
    

    for i, evaluation_resample in tqdm(sorted(dataset_split['evaluation_resamples'].items()), desc='evaluation_resample'):
        resample_rng = random.Random(rng.randint(0, 2^31))
        tag = f'{i}'
        try:
            nested_resample_tracker = experiment_tracker.get_child(tag, "evaluation_resample")
        except KeyError:
            nested_resample_tracker = experiment_tracker.make_child("evaluation_resample", tag=tag)
        try:
            if nested_resample_tracker.check_flag("work_is_done"):
                continue
        except FileNotFoundError:
            nested_resample_tracker.set_flag("work_is_done", False)
        test_dataset_manager = dataset.resample_from_summary(evaluation_resample['test'])
        dev_dataset_manager = dataset.resample_from_summary(evaluation_resample['dev'])
        train_dataset_manager = dataset.resample_from_summary(evaluation_resample['train'])
        
        test_dataset = test_dataset_manager.instantiate_test_dataset()

        train_dataset = train_dataset_manager.instantiate_training_dataset()
        dev_dataset = dev_dataset_manager.instantiate_test_dataset()

        nested_resample_tracker.log_json("test_dataset", train_dataset_manager.summarize())
        nested_resample_tracker.log_json("train_dataset", train_dataset_manager.summarize())
        nested_resample_tracker.log_json("dev_dataset", dev_dataset_manager.summarize())
        
        model = run_training(train_dataset, dev_dataset, best_config, nested_resample_tracker, rng=resample_rng)
        
        dev_performance, dev_predictions = model.evaluate_dataset(dev_dataset)
        nested_resample_tracker.log_performance(dataset_name="dev_dataset", values=dev_performance)
        nested_resample_tracker.log_table("dev_predictions", dev_predictions)
        
        test_performance, sample_predictions = model.evaluate_dataset(test_dataset)
        nested_resample_tracker.log_performance(dataset_name='test_dataset', values=test_performance)
        nested_resample_tracker.log_table("test_predictions", sample_predictions)
        nested_resample_tracker.set_flag("work_is_done")
    experiment_tracker.set_flag("work_is_done")


def reevaluate_resample(experiment_tracker_dir: Path):
    experiment_tracker = ExperimentTracker(experiment_tracker_dir)
    try:
        work_package = experiment_tracker.load_artifact('work_package')
    except FileNotFoundError:
        print(f"No workpackage in experiment {experiment_tracker_dir}")
        return
    
    heldout_dataset_manager =  work_package['heldout_dataset']
    dataset_tag = work_package['dataset_tag']
    
    try:
        heldout_dataset = heldout_dataset_manager.instantiate_test_dataset()
        best_model_reference = experiment_tracker.lookup_reference('best_model')
        best_model = experiment_tracker.load_model(best_model_reference)
        test_performance, sample_predictions = best_model.evaluate_dataset(heldout_dataset)
        experiment_tracker.log_performance(dataset_name=dataset_tag, values=test_performance)
        experiment_tracker.log_json("test_dataset", heldout_dataset_manager.summarize())
        experiment_tracker.log_table("test_predictions", sample_predictions)
    except FileNotFoundError:
        print(f"No models in the root of {experiment_tracker_dir}")

    for child_tracker in experiment_tracker.get_children():
        # The children don't have work packages, let's be smart about how to do this instead
        try:
            best_model_reference = child_tracker.lookup_reference('best_model')
            best_model = child_tracker.load_model(best_model_reference)
            test_performance, sample_predictions = best_model.evaluate_dataset(heldout_dataset)
            child_tracker.log_performance(dataset_name=dataset_tag, values=test_performance)
            child_tracker.log_json("test_dataset", heldout_dataset_manager.summarize())
            child_tracker.log_table("test_predictions", sample_predictions)
        except FileNotFoundError:
            print(f"No models in child experiment {child_tracker.output_dir}")

            
    
 

    # In the future, the runs will be nested. Think about how to handle both the legacy case above and the new case with nested evaluations
    # for train_dataset_manager, dev_dataset_manager in fitting_dataset.resample(experiment_config.nested_resample_strategy):
    #     nested_resample_tracker = experiment_tracker.make_child()
    #     train_dataset = train_dataset_manager.instantiate_training_dataset()
    #     dev_dataset = dev_dataset_manager.instantiate_test_dataset()
    #     nested_resample_tracker.log_json("training_dataset", train_dataset_manager.summarize())
    #     nested_resample_tracker.log_json("dev_dataset", dev_dataset_manager.summarize())
    #     model = run_training(train_dataset, dev_dataset, best_config, nested_resample_tracker, rng=rng)
        
        
    
        
    # experiment_tracker.set_flag("work_is_done")

def run_experiment(config_path: Path, output_dir: Path):
    config = load_object(config_path, ExperimentConfig)
    rng = random.Random(config.random_seed)
    name = config.name
    
    experiment_directory = output_dir / name / timestamp()
    
    with ExperimentTracker(experiment_directory) as experiment_tracker:
        experiment_tracker.log_artifact('experiment_config', config)
        experiment_tracker.log_file(config_path, 'experiment_config.py')
    
    dataset = load_dataset(config.dataset_config_path)
    dataset_splits = load_dataset_splits(config_path)

    
    resample_directories = []
    for i, resample_split in dataset_splits.items():
        fold_experiment_tracker = experiment_tracker.make_child()
        random_seed = rng.randint(0, 2**31)
        resample_directory = setup_work_package(experiment_tracker=fold_experiment_tracker, 
                                                experiment_config=config,
                                                dataset=dataset,
                                                dataset_splits=resample_split,
                                                random_seed=random_seed)
        resample_directories.append(resample_directory)
    
    process_work_packages(config=config, work_package_directories=resample_directories)


def process_work_packages(config: ExperimentConfig, work_package_directories, inject_config = False):
    if inject_config:
        for resample_experiment_dir in work_package_directories:
            resample_tracker = ExperimentTracker(resample_experiment_dir)
            work_package = resample_tracker.load_artifact('work_package')
            work_package['experiment_config'] = config
            resample_tracker.log_artifact('work_package', work_package)


    if config.multiprocessing_envs:
        # We will use spawn so that we can control the environment better
        if hasattr(multiprocessing, 'get_context'):
            context = multiprocessing.get_context('spawn') 
        else:
            context = multiprocessing
        work_queue = context.Queue()
        for resample_experiment_dir in work_package_directories:
            work_queue.put(resample_experiment_dir)
        results_queue = context.Queue() 
        processes = [context.Process(target=resample_worker, args=(work_queue, results_queue, environment)) for environment in config.multiprocessing_envs]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    else:
        for resample_experiment_dir in tqdm(work_package_directories, 'work_package'):
            experiment_on_resample(resample_experiment_dir)

def hp_run(config: ExperimentConfig, model_config: ModelWrapperConfig, dataset: AbstractDataset, dataset_split, trial: optuna.Trial, experiment_tracker: ExperimentTracker, rng: random.Random, rerun_search: bool = False):
    trial_tag = f'{trial.number:02}'
    hp_run_tracker = experiment_tracker.make_child(tag=trial_tag, child_group='hpo_trials')
    instantiated_model_config = suggest_values(model_config, trial)
    hp_run_tracker.log_artifact("model_config", instantiated_model_config)

    hp_run_performance_collection = []
    for i, root_resample_split in dataset_split.items():
        for j, nested_resample_split in root_resample_split.items():
            tag = f"{i}_{j}"
            hp_fold_tracker = hp_run_tracker.make_child(tag=tag)
            
            test_dataset_manager = dataset.resample_from_summary(nested_resample_split['test'])
            dev_dataset_manager = dataset.resample_from_summary(nested_resample_split['dev'])
            train_dataset_manager = dataset.resample_from_summary(nested_resample_split['train'])
            
            test_dataset = test_dataset_manager.instantiate_test_dataset()
            dev_dataset = dev_dataset_manager.instantiate_test_dataset()
            train_dataset = train_dataset_manager.instantiate_training_dataset()
            
            hp_fold_tracker.log_json("test_dataset", test_dataset_manager.summarize())
            hp_fold_tracker.log_json("training_dataset", train_dataset_manager.summarize())
            hp_fold_tracker.log_json("dev_dataset", dev_dataset_manager.summarize())
            model = run_training(train_dataset, dev_dataset, instantiated_model_config, hp_fold_tracker, rng=rng)
            
            subset_performance_dict, predictions = model.evaluate_dataset(test_dataset)
            hp_run_performance_collection.append(subset_performance_dict[config.hpo_config.hp_optimization_metric])
            hp_fold_tracker.log_performance(dataset_name="test_dataset", values=subset_performance_dict)
            hp_fold_tracker.log_table("test_predictions", predictions)
            if not config.hpo_config.save_model:
                hp_fold_tracker.purge_models()
        
    hp_run_performance = aggregate(hp_run_performance_collection, config.hpo_config.aggregation_method)
    if not config.hpo_config.save_model:
        hp_run_tracker.purge_models()
        
    return hp_run_performance


def aggregate(data, method=Literal['mean', 'median', 'min', 'max', 'stdev']):
    if method == 'mean':
        return statistics.mean(data)
    elif method == 'median':
        return statistics.median(data)
    elif method == 'stdev':
        return statistics.stdev(data)
    elif method == 'max':
        return max(data)
    elif method == 'min':
        return min(data)
    else:
        raise ValueError(f"Not supported aggregation method: '{method}'")


def hpo_search(experiment_config: ExperimentConfig, dataset: AbstractDataset, dataset_split, experiment_tracker: ExperimentTracker, rng: random.Random, rerun_hpo_search=False):
    hp_config = experiment_config.hpo_config
    model_config = load_object(experiment_config.model_config_path, ModelWrapperConfig)
    
    sampler_seed = rng.randint(0, 2**32-1)
    n_trials = hp_config.hp_iterations
    try:
        study = experiment_tracker.load_artifact('optuna_study')
        n_trials = n_trials - len(study.get_trials())
    except FileNotFoundError:    
        sampler = get_sampler(hp_config.sampler, sampler_seed)
        hp_direction = experiment_config.hpo_config.hp_direction
        study = create_study(direction=hp_direction, sampler=sampler)

    if n_trials > 0:
        for hp_run_tracker in experiment_tracker.get_children(child_group='hpo_trials'):
            try:
                instantiated_model_config = hp_run_tracker.load_artifact('model_config')
                params, distributions = get_hp_params(model_config, instantiated_model_config)
                ## Figure out the performance value for this trial
                values = []
                for hp_fold_run in hp_run_tracker.get_children():
                    test_performance_path = hp_fold_run.output_dir / 'evaluation' / 'test_dataset' / 'performance' / 'performance.csv'
                    if test_performance_path.exists():
                        with open(test_performance_path) as fp:
                            lines = [line.strip().split(',') for line in fp]
                            header = lines[0]
                            columns = {fieldname: list() for fieldname in header}
                            for line in lines[1:]:
                                for fieldname, value in zip(header, line):
                                    columns[fieldname].append(value)
                            value = columns[hp_config.hp_optimization_metric][-1]  # Here we assume the last value is the valid one
                            values.append(float(value)) 
                value = aggregate(values, hp_config.aggregation_method)
                # add trial to study here
                trial = optuna.trial.create_trial(value=value, params=params, distributions=distributions)
                study.add_trial(trial)
                n_trials -= 1
            except:
                continue
        

        hp_run_bound = partial(hp_run, experiment_config, model_config, dataset, dataset_split, experiment_tracker=experiment_tracker, rng=rng, rerun_search=rerun_hpo_search)
        study.optimize(hp_run_bound, n_trials=n_trials)

        experiment_tracker.log_artifact('optuna_study', study)
    
    finalized_config = best_values(model_config, study)
    return finalized_config


def get_hpo_results(experiment_config: ExperimentConfig, experiment_tracker: ExperimentTracker):
    model_config = load_object(experiment_config.model_config_path, ModelWrapperConfig)
    study = experiment_tracker.load_artifact('optuna_study')
    finalized_config = best_values(model_config, study)
    return finalized_config



   
def find_pending_resamples(experiment_directory: Path):
    "Finds all experiment subdirectories which don't have the experiment done flag set"
    # Perhaps not the best way of ensuring we get the root experiments is to check common prefixes for all folders with
    # a experiment_metadata subdirectory
    
    experiments_paths = find_experiments(experiment_directory)
    unfinished_experiments = []
    
    for p in experiments_paths:
        tracker = ExperimentTracker(p)
        # Only the resamples has work packages, this filters out HPO trials
        try:
            work_package = tracker.load_artifact('work_package')
        except FileNotFoundError:
            continue
        
        if not (p / 'evaluation').exists():
            unfinished_experiments.append(p)
            continue
            
        # try:
        #     if not tracker.check_flag('work_is_done'):
        #         unfinished_experiments.append(p)
        # except FileNotFoundError:  # Old code used the existance of a file to flag wheter it was set, we account for this
        #     unfinished_experiments.append(p)
    
    return unfinished_experiments

def continue_experiments(config: ExperimentConfig, experiment_directories, rerun_hpo_search=False, replace_config=True):
    for experiment_directory in experiment_directories:
        tracker = ExperimentTracker(experiment_directory)
        work_package = tracker.load_artifact('work_package')
        if rerun_hpo_search:
            work_package['rerun_status'] = 'full'
        else:
            work_package['rerun_status'] = 'limited'
        if replace_config:
            work_package['experiment_config'] = config
        tracker.log_artifact('work_package', work_package)
    
    process_work_packages(config, experiment_directories)
    


def rerun_experiments(config: ExperimentConfig, experiment_directories, rerun_hpo_search=False, replace_config=True):
    for experiment_directory in experiment_directories:
        tracker = ExperimentTracker(experiment_directory)
        work_package = tracker.load_artifact('work_package')
        if rerun_hpo_search:
            work_package['rerun_status'] = 'full'
        else:
            work_package['rerun_status'] = 'limited'
        if replace_config:
            work_package['experiment_config'] = config
        tracker.log_artifact('work_package', work_package)
    
    process_work_packages(config, experiment_directories)
    
    

def reevaluate_experiments(experiment_directories):
    for experiment_directory in experiment_directories:
        reevaluate_resample(experiment_directory)


def setup_data_splits(config_path: Path):
    config = load_object(config_path, ExperimentConfig)
    rng = random.Random(config.random_seed)
    
    dataset = load_dataset(config.dataset_config_path)
    #dataset_config = load_object(config.dataset_config_path, DatasetConfig)
    #dataset = load_dataset(dataset_config)
    root_resamples = dataset.resample(config.resample_strategy, rng)
    
    dataset_splits = {}
    for i, (modeling_dataset_manager, heldout_dataset_manager) in enumerate(root_resamples):
        test_split = heldout_dataset_manager.summarize()
        
        hpo_resamples = defaultdict(dict)
        
        for hpo_root_i, (hpo_modeling_dataset_manager, hpo_test_dataset_manager) in enumerate(modeling_dataset_manager.resample(config.hpo_config.root_resample_strategy, rng)):
            hpo_test_dataset = hpo_test_dataset_manager.summarize()
            for hpo_resample_j, (hpo_train_dataset_manager, hpo_dev_dataset_manager) in enumerate(hpo_modeling_dataset_manager.resample(config.hpo_config.nested_resample_strategy, rng)):
                hpo_train_dataset = hpo_train_dataset_manager.summarize()
                hpo_dev_dataset = hpo_dev_dataset_manager.summarize()
                hpo_resamples[hpo_root_i][hpo_resample_j] = {'test': hpo_test_dataset, 'dev': hpo_dev_dataset, 'train': hpo_train_dataset}
        
        evaluation_resamples = dict()
        for evaluation_i, (evaluation_train_dataset_manager, evaluation_dev_dataset_manager) in enumerate(modeling_dataset_manager.resample(config.nested_resample_strategy, rng)):
            evaluation_training_dataset = evaluation_train_dataset_manager.summarize()
            evaluation_dev_dataset = evaluation_dev_dataset_manager.summarize()
            evaluation_resamples[evaluation_i] = {'test': test_split, 'train': evaluation_training_dataset, 'dev': evaluation_dev_dataset}
        
        dataset_splits[i] = {'test_split': test_split, 'hpo_resamples': hpo_resamples, 'evaluation_resamples': evaluation_resamples}
    
    dataset_split_path = Path(config.data_split_path)
    if dataset_split_path is None:
        config_name = config_path.with_suffix('').name
        dataset_split_path = config_path.parent / 'dataset_splits' / f"{config_name}.json"
    dataset_split_path.parent.mkdir(exist_ok=True, parents=True)
    with open(dataset_split_path, 'w') as fp:
        json.dump(dataset_splits, fp, indent=2, sort_keys=True)
    return dataset_splits


def load_dataset_splits(config_path):
    config = load_object(config_path, ExperimentConfig)
    dataset_split_path = config.data_split_path
    if dataset_split_path is None:
        config_name = config_path.with_suffix('').name
        dataset_split_path = config_path.parent / 'dataset_splits' / f"{config_name}.json"
    with open(dataset_split_path) as fp:
        dataset_splits = json.load(fp)
    return dataset_splits

