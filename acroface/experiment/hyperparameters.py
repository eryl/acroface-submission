
import argparse
from cmath import exp
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
import types
from typing import Sequence, Mapping, Type, Dict, Any, Optional, List, Union, Literal, Collection, Set

import numpy as np
import optuna

class HyperParameterError(Exception):
    def __init__(self, obj, message):
        self.obj = obj
        self.message = message

    def __str__(self):
        return f"{self.message}: {self.obj}"


class HyperParameter(object):
    def __init__(self, *, name: str):
        self.name = name
        self.trial_values = dict()

    def get_value(self, trial_or_study: Union[optuna.Trial, optuna.Study]):
        if isinstance(trial_or_study, optuna.Trial):
            return self.get_trial_value(trial_or_study)
        elif isinstance(trial_or_study, optuna.Study):
            return self.get_best_value(trial_or_study)
        else:
            raise ValueError(f"Can't get value with context object {trial_or_study}")

    def get_trial_value(self, trial: optuna.Trial):
        # It seems like we would not have to do this, since a trial will return the same value for the same parameter,
        # on the other hand this makes the framework robust to other implementations. Using the datetime start should be
        # more robust trials belonging to different studies (as will the outer cross validation loop)
        trial_id = trial.datetime_start
        if trial_id not in self.trial_values:
            self.trial_values[trial_id] = self.suggest_value(trial)
        else:
            pass
        return self.trial_values[trial_id]

    def get_best_value(self, study: optuna.Study):
        return study.best_params[self.name]

    def suggest_value(self, trial: optuna.Trial):
        raise NotImplementedError("Can not suggest value for base class HyperParameter")

    def set_value(self, value, trial: optuna.Trial):
        pass
    
    def get_hp_param(self, value):
        distribution = self.get_distribution()
        return {'name': self.name, 'param': value, 'distribution': distribution}


class HyperParameterCatergorical(HyperParameter):
    def __init__(self, *, choices: Sequence[Any], **kwargs):
        super(HyperParameterCatergorical, self).__init__(**kwargs)
        self.choices = choices

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_categorical(self.name, self.choices)

    def get_distribution(self):
        return optuna.distributions.CategoricalDistribution(choices=self.choices)

class HyperParameterDiscreteUniform(HyperParameter):
    def __init__(self, *, low:  float, high: float, q: int, **kwargs):
        super(HyperParameterDiscreteUniform, self).__init__(**kwargs)
        self.low = low
        self.high = high
        self.q = q

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_discrete_uniform(self.name, self.low, self.high, self.q)

    def get_distribution(self):
        return optuna.distributions.FloatDistribution(self.low, self.high, log=False, step=self.q)

class HyperParameterFloat(HyperParameter):
    def __init__(self, *, low:  float,
                 high: float,
                 step: Optional[float] = None,
                 log: Optional[bool] = False,
                 **kwargs):
        super(HyperParameterFloat, self).__init__(**kwargs)
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)
    
    def get_distribution(self):
        return optuna.distributions.FloatDistribution(self.low, self.high, log=self.log, step=self.step)


class HyperParameterInteger(HyperParameter):
    def __init__(self, *, low: int,
                 high: int,
                 step: Optional[int] = 1,
                 log: Optional[bool] = False,
                 **kwargs):
        super(HyperParameterInteger, self).__init__(**kwargs)
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_int(self.name, self.low, self.high, step=self.step, log=self.log)

    def get_distribution(self):
        return optuna.distributions.IntDistribution(low=self.low, high=self.high, log=self.log, step=self.step)


class HyperParameterLogUniform(HyperParameter):
    def __init__(self, *, low: float, high: float, **kwargs):
        super(HyperParameterLogUniform, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_loguniform(self.name, self.low, self.high)

    def get_distribution(self):
        return optuna.distributions.FloatDistribution(self.low, self.high, log=True)
    

class HyperParameterUniform(HyperParameter):
    def __init__(self, *, low: float, high: float, **kwargs):
        super(HyperParameterUniform, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_uniform(self.name, self.low, self.high)

    def get_distribution(self):
        return optuna.distributions.FloatDistribution(self.low, self.high)
    
    
class HyperParameterFunction(HyperParameter):
    def __init__(self, *, function, **kwargs):
        super(HyperParameterFunction, self).__init__(**kwargs)
        self.function = function

    def get_value(self, trial_or_study):
        return self.function(trial_or_study)


def instantiate_hp_value(obj, trial_or_study: Union[optuna.Trial, optuna.Study]):
    non_collection_types = (str, bytes, bytearray, np.ndarray)
    try:
        if isinstance(obj, (type, types.FunctionType, types.LambdaType, types.ModuleType)):
            return obj
        if isinstance(obj, HyperParameter):
            return obj.get_value(trial_or_study)
        elif isinstance(obj, Mapping):
            return type(obj)({k: instantiate_hp_value(v, trial_or_study) for k, v in obj.items()})
        elif isinstance(obj, Collection) and not isinstance(obj, non_collection_types):
            return type(obj)(instantiate_hp_value(x, trial_or_study) for x in obj)
        elif hasattr(obj, '__dict__'):
            try:
                obj_copy = copy.copy(obj)
                obj_copy.__dict__ = instantiate_hp_value(obj.__dict__, trial_or_study)
                return obj_copy
            except TypeError:
                return obj
        else:
            return obj
    except TypeError as e:
        raise HyperParameterError(obj, "Failed to materialize") from e


def set_hp_value(obj, reference_obj, trial_or_study: Union[optuna.Trial, optuna.Study]):
    non_collection_types = (str, bytes, bytearray, np.ndarray)
    try:
        if isinstance(obj, (type, types.FunctionType, types.LambdaType, types.ModuleType)):
            return obj
        if isinstance(obj, HyperParameter):
            return obj.set_value(reference_obj, trial_or_study)
        elif isinstance(obj, Mapping):
            return type(obj)({k: set_hp_value(v, reference_obj[k], trial_or_study) for k, v in obj.items()})
        elif isinstance(obj, Sequence) and not isinstance(obj, non_collection_types):
            return type(obj)(set_hp_value(x, x_ref, trial_or_study) for x, x_ref in zip(obj))
        elif hasattr(obj, '__dict__'):
            try:
                obj_copy = copy.copy(obj)
                obj_copy.__dict__ = set_hp_value(obj.__dict__, reference_obj.__dict__, trial_or_study)
                return obj_copy
            except TypeError:
                return obj
        elif isinstance(obj, Set):
            raise RuntimeError(f"Set objects can't be properly instantiated:{obj}")
        else:
            return obj
    except TypeError as e:
        raise HyperParameterError(obj, "Failed to materialize") from e


def get_hp_params_(obj, reference_obj, params, distributions):
    non_collection_types = (str, bytes, bytearray, np.ndarray)
    try:
        if isinstance(obj, (type, types.FunctionType, types.LambdaType, types.ModuleType)):
            return obj
        if isinstance(obj, HyperParameter):
            param_info = obj.get_hp_param(reference_obj)
            name = param_info['name']
            params[name] = param_info['param']
            distributions[name] = param_info['distribution']
            
        elif isinstance(obj, Mapping):
            return type(obj)({k: get_hp_params_(v, reference_obj[k], params, distributions) for k, v in obj.items()})
        elif isinstance(obj, Sequence) and not isinstance(obj, non_collection_types):
            return type(obj)(get_hp_params_(x, x_ref, params, distributions) for x, x_ref in zip(obj, reference_obj))
        elif hasattr(obj, '__dict__'):
            try:
                obj_copy = copy.copy(obj)
                obj_copy.__dict__ = get_hp_params_(obj.__dict__, reference_obj.__dict__, params, distributions)
                return obj_copy
            except TypeError:
                return obj
        elif isinstance(obj, Set):
            raise RuntimeError(f"Set objects can't be properly instantiated:{obj}")
        else:
            return obj
    except TypeError as e:
        raise HyperParameterError(obj, "Failed to materialize") from e

def get_hp_params(obj, reference_obj):
    params = dict()
    distributions = dict()
    get_hp_params_(obj, reference_obj, params, distributions)
    return params, distributions


def suggest_values(hp_object, trial: optuna.Trial):
    return instantiate_hp_value(hp_object, trial)


def best_values(hp_object, study):
    return instantiate_hp_value(hp_object, study)


def set_values(hp_object, reference_object, trial: optuna.Trial):
    return set_hp_value(hp_object, reference_object, trial)


def get_sampler(method, random_seed):
    if method == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=random_seed)
    elif method == 'random':
        sampler = optuna.samplers.RandomSampler(seed=random_seed)
    else:
        raise NotImplementedError(f"Sample strategy {method} has not been implemented")
    return sampler
   


def create_study(direction, sampler):
    return optuna.create_study(direction=direction, sampler=sampler)