from typing import Union, Literal, Optional
from dataclasses import dataclass

class ResampleStrategy:
    def __init__(self) -> None:
        pass


@dataclass
class KFoldStrategy(ResampleStrategy):
    k: Union[int, Literal['n', 'N']]
    stratified: bool = False
    n_runs: int = 1
    subset: Optional[int] = None  # If set, only return this many total folds 

@dataclass
class GroupKFoldStrategy(ResampleStrategy):
    k: Union[int, Literal['n', 'N']]
    stratified: bool = False
    n_runs: int = 1
    subset: Optional[int] = None  # If set, only return this many total folds 

@dataclass
class GroupSubsampleStrategy(ResampleStrategy):
    n_runs: int = 1
    test_size: float = 0.1
    

class DefaultStrategy(ResampleStrategy):
    pass

    