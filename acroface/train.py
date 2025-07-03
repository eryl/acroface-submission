import random
from acroface.dataset.core import AbstractDataset
from acroface.models.wrapper import ModelWrapperConfig, load_model
from acroface.experiment.tracker import ExperimentTracker



def run_training(train_dataset: AbstractDataset, 
                 dev_dataset: AbstractDataset, 
                 instantiated_model_config: ModelWrapperConfig, 
                 hp_run_tracker: ExperimentTracker,
                 rng: random.Random):
    model = load_model(instantiated_model_config, rng)
    model.fit(train_dataset=train_dataset, dev_dataset=dev_dataset, experiment_tracker=hp_run_tracker)
    return model
