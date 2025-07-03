from typing import Type, Union, TypeVar, Optional
from pathlib import Path
from dataclasses import dataclass, field
import random
from copy import copy, deepcopy
from typing import Tuple, Dict, List

from acroface.dataset.resample import ResampleStrategy, DefaultStrategy, KFoldStrategy, GroupKFoldStrategy, GroupSubsampleStrategy
from acroface.util import load_object


class DatasetView:
    def __init__(self) -> None:
        pass    
    
    def __getitem__(self, item):
        raise NotImplementedError()

    def summarize(self):
        raise NotImplementedError()
    
    def get_collate_fn(self):
        raise NotImplementedError()
    

@dataclass
class DatasetConfig:
    dataset_class: Type["AbstractDataset"]  # This should be some other class
    dataset_args: Tuple = field(default_factory=tuple)    
    dataset_kwargs: Dict = field(default_factory=dict)
    

ClassVar = TypeVar('ClassVar')
class AbstractDataset:
    def __init__(self, indices=None):
        if indices is None:
            indices = self.get_index()
        self.indices = indices
        self.labels = self.get_labels()
        self.groups = self.get_groups()
        self.name = "unamed_dataset"
    
    def __len__(self):
        return len(self.indices)
    
    def make_subset(self, indices=None):
        raise NotImplementedError(f"make_subset() has not been implemented for {type(self)}")
    
    def get_index(self) -> List[int]:
        raise NotImplementedError(f"get_index() has not been implemented for {type(self)}")
    
    def get_labels(self):
        raise NotImplementedError(f"get_labels() has not been implemented for {type(self)}")
    
    def get_groups(self, index=None):
        raise NotImplementedError(f"get_groups() has not been implemented for {type(self)}")
    
    @classmethod
    def from_config(cls: ClassVar, dataset_config: DatasetConfig) -> ClassVar:
        # Find the common index between the different datasets
        dataset = dataset_config.dataset_class(*dataset_config.datset_args, **dataset_config.dataset_kwargs)
        return cls(dataset)
        
    def resample(self, resample_strategy: ResampleStrategy, rng) -> "AbstractDataset":
        if isinstance(resample_strategy, DefaultStrategy):
            yield self.default_split()
        
        elif isinstance(resample_strategy, GroupSubsampleStrategy):
            from sklearn.model_selection import GroupShuffleSplit
            seed = rng
            if rng is not None:
                seed = rng.randint(0, 2**32-1)
            resampler = GroupShuffleSplit(n_splits=resample_strategy.n_runs, test_size=resample_strategy.test_size, random_state=seed)
            labels = self.get_labels()
            groups = self.get_groups()
            for i, (fit_meta_indices, test_meta_indices) in enumerate(resampler.split(self.indices, y=labels, groups=groups)):
                # indices above are not values of self.indices, but instead the indices into this
                fit_indices = [self.indices[i] for i in fit_meta_indices]
                test_indices = [self.indices[i] for i in test_meta_indices]
                fit_dataset = self.make_subset(indices=fit_indices)
                test_dataset = self.make_subset(indices=test_indices)
                yield fit_dataset, test_dataset
    
        elif isinstance(resample_strategy, KFoldStrategy):
            from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut
            seed = rng
            if rng is not None:
                seed = rng.randint(0, 2**32-1)
            if resample_strategy.k == 'n' or resample_strategy.k == 'N':
                resampler = LeaveOneOut()
            elif resample_strategy.stratified:
                resampler = RepeatedStratifiedKFold(n_splits=resample_strategy.k, n_repeats=resample_strategy.n_runs, random_state=seed)
            else:
                resampler = RepeatedKFold(n_splits=resample_strategy.k, n_repeats=resample_strategy.n_runs, random_state=seed)

            for i, (fit_meta_indices, test_meta_indices) in enumerate(resampler.split(self.indices, y=labels, groups=groups)):
                if resample_strategy.subset is not None and resample_strategy.subset <= i:
                        break
                # indices above are not values of self.indices, but instead the indices into this
                fit_indices = [self.indices[i] for i in fit_meta_indices]
                test_indices = [self.indices[i] for i in test_meta_indices]
                fit_dataset = self.make_subset(indices=fit_indices)
                test_dataset = self.make_subset(indices=test_indices)
                yield fit_dataset, test_dataset
                
        elif isinstance(resample_strategy, GroupKFoldStrategy):
            from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
            seed = rng
            if rng is not None:
                seed = rng.randint(0, 2**32-1)
            if resample_strategy.k == 'n' or resample_strategy.k == 'N':
                resampler = LeaveOneGroupOut()
            elif resample_strategy.stratified:
                resampler = StratifiedGroupKFold(n_splits=resample_strategy.k)
            else:
                resampler = GroupKFold(n_splits=resample_strategy.k)
            
            labels = self.get_labels()
            groups = self.get_groups()
            for repeat in range(resample_strategy.n_runs):
                for i, (fit_meta_indices, test_meta_indices) in enumerate(resampler.split(self.indices, y=labels, groups=groups)):
                    if resample_strategy.subset is not None and resample_strategy.subset <= i:
                        break
                    # indices above are not values of self.indices, but instead the indices into this
                    fit_indices = [self.indices[i] for i in fit_meta_indices]
                    test_indices = [self.indices[i] for i in test_meta_indices]
                    fit_dataset = self.make_subset(indices=fit_indices)
                    test_dataset = self.make_subset(indices=test_indices)
                    yield fit_dataset, test_dataset
        # elif resample_strategy == None:
        #     print("Using default resample strategy")
            
        else:
            raise NotImplementedError(f"Resampling strategy {type(resample_strategy)} has not been implemented")

    def default_split(self) -> "AbstractDataset":
        """
        Return the default trainval/test split of this dataset if there is one"""
        raise NotImplementedError(f'default_split() has not been implemented for {cls}')

    def instantiate_training_dataset(self) -> DatasetView:
        raise NotImplementedError(f"instantiate_training_dataset() has not been implemented for {type(self)}")
    
    def instantiate_test_dataset(self) -> DatasetView: 
        raise NotImplementedError(f"instantiate_test_dataset() has not been implemented for {type(self)}")        

    def summarize(self):
        return dict(groups=self.groups, indices=self.indices, labels=self.labels)
    
    def resample_from_summary(self, summary):
        indices = summary['indices']
        return self.make_subset(indices)
        

def load_dataset(dataset_config_path: Path) -> AbstractDataset:
    dataset_config = load_object(dataset_config_path, DatasetConfig)
    dataset = dataset_config.dataset_class(*dataset_config.dataset_args, **dataset_config.dataset_kwargs)
    return dataset

