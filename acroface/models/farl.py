from pathlib import Path

from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.model_zoo import load_url
import torchvision
from torchvision.io import read_image
from torchvision.transforms import v2


from acroface.dataset.core import AbstractDataset

import clip

from acroface.models.torch_model import TorchModelConfig, TorchModel

FARL_URL = "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"


class FarlModule(torch.nn.Module):
    def __init__(self, clip_visual, fc):
        super().__init__()
        self.clip_visual = clip_visual
        self.fc = fc
        
    def forward(self, image):
        dtype = self.clip_visual.conv1.weight.dtype
        cast_image = image.type(dtype)
        representations = self.clip_visual(cast_image)
        logits = self.fc(representations)
        return logits
        
    

class FarlModelConfig(TorchModelConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _transform(n_px):
    # We're using non-public interfaces here, risk for screw up
    from torchvision.transforms._presets import ImageClassification
    from torchvision.transforms.functional import InterpolationMode
    transform = ImageClassification(crop_size=n_px, resize_size=n_px, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711),
                                    interpolation=InterpolationMode.BICUBIC)
    return transform


class FarlModel(TorchModel):
    def __init__(self, *args, config: TorchModelConfig, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        
    def setup_initialization_params(self, train_dataset, model_path=Path("pretrained_models/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth")):
        #farl_state = torch.load("pretrained_models/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
        model_path.parent.mkdir(exist_ok=True, parents=True)
        farl_state = load_url(FARL_URL, model_dir=model_path.parent, file_name=model_path.name)
        self.initialization_params['weights'] = farl_state['state_dict']
        super().setup_initialization_params(train_dataset)
    
    def initialize_network(self):
        weights = self.initialization_params['weights']
        clip_model, _clip_preprocess = clip.load("ViT-B/16", device="cpu")
        clip_model.load_state_dict(weights, strict=False)

        self.visual = clip_model.visual
        self.preprocess = _transform(self.visual.input_resolution)

        self.train_transforms = v2.Compose([v2.AutoAugment(), self.preprocess])
        self.fc = torch.nn.Sequential(torch.nn.Linear(self.visual.output_dim, self.config.hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(self.config.dropout_rate), torch.nn.Linear(self.config.hidden_dim, 1))
        self.model = FarlModule(self.visual, self.fc)
       
        #self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
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
        return self.model.fc.parameters()
       
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
        
    