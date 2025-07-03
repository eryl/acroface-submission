from typing import TypeVar, Type, Sequence, Mapping, Any
from dataclasses import dataclass, field
import random
from pathlib import Path

from acroface.util import load_object

ClassVar = TypeVar('ClassVar')

@dataclass
class ModelWrapperConfig:
    model_class: Type
    model_args: Sequence = field(default_factory=tuple)
    model_kwargs: Mapping = field(default_factory=dict)


class ModelWrapper:
    def __init__(self, model_object: Any, model_config: ModelWrapperConfig):
        self.model_object = model_object
        self.model_config = model_config

    @classmethod
    def from_config(cls: Type[ClassVar], config: ModelWrapperConfig, rng: random.Random) -> ClassVar:
        return cls(model_object, config)
    
    def fit(self, train_dataset, dev_dataset, experiment_tracker):
        self.model_object.fit(train_dataset=train_dataset, dev_dataset=dev_dataset, experiment_tracker=experiment_tracker)

    def predict_dataset(self, dataset):
        return self.model_object.predict_dataset(dataset)

    def evaluate_dataset(self, dataset):
        predictions = self.predict_dataset(dataset)
        pass
    
def load_model(config: ModelWrapperConfig, rng):
    model_object = config.model_class(*config.model_args, **config.model_kwargs, rng=rng)
    return model_object