

from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import torchvision
from torchvision.io import read_image
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2
from acroface.dataset.core import AbstractDataset




from acroface.models.torch_model import TorchModelConfig, TorchModel

        

class DenseNetModelConfig(TorchModelConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DenseNetModel(TorchModel):
    def __init__(self, *args, config: TorchModelConfig, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        
    def setup_initialization_params(self, train_dataset):
        self.initialization_params['weights'] = DenseNet121_Weights.DEFAULT
        super().setup_initialization_params(train_dataset)
    
    def initialize_network(self):
        weights = self.initialization_params['weights']
        self.preprocess = weights.transforms()
        self.train_transforms = v2.Compose([v2.AutoAugment(), self.preprocess])
        
        self.model = densenet121(weights=weights)
        #self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 1)
        self.model.classifier = torch.nn.Sequential(torch.nn.Linear(self.model.classifier.in_features, self.config.hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(self.config.dropout_rate), torch.nn.Linear(self.config.hidden_dim, 1))
        self.model.to(self.device)

        params = []
        # if self.config.train_encoder:
        #     params.extend(self.encoder.parameters())
        # else:
        #     for p in self.encoder.parameters():
        #         p.requires_grad = False

        params.extend(self.model.parameters())
        self.params = params
    
    def get_frozen_pretraining_params(self):
        return self.model.classifier.parameters()

    def setup_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()
    
    def loss_on_batch(self, batch):
        x = batch['data']
        y = batch['label']
        x = x.to(self.device)
        y = y.to(self.device)
        pred = self.model(x)
        loss = self.loss(pred, y)
        return loss, y, pred
        
    