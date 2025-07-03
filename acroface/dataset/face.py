import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import re
from collections import defaultdict, Counter
import random

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision.io import read_image, decode_image
import numpy as np
from numpy.lib.format import open_memmap

import cryptography
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from acroface.dataset.core import DatasetConfig, AbstractDataset, DatasetView

CACHE_DIR_SUFFIX = ".npy"


@dataclass
class FaceDatasetConfig:
    name: str
    root_directory: Path
    training_subdir: str
    test_subdir: str
    file_extensions: Sequence[str] = ('png', 'jpg', 'jpeg')
    data_dir_prefix: str = "masked_yaws"
    subject_index_file: str = "subjects.txt"


def get_file_info(acroface_file):
    pattern = r"([A-Za-z]+)(\d+)-(A|K)-?.*"
    m = re.match(pattern, acroface_file)
    if m is not None:
        site, study_id, label = m.groups()
        group = f"{site}{study_id}"
        subject_id = f"{group}-{label}"
        return {'label': label, 'group': group, 'site': site, 'study_id': study_id, 'subject_id': subject_id}
    else:
        raise RuntimeError(f"Can't match {acroface_file}")
    
        
class FaceDatasetView(Dataset, DatasetView):
    def __init__(self, config: FaceDatasetConfig, base_dir: Path, subjects):
        self.base_dir = base_dir
        self.name = config.name
        subject_images = defaultdict(list)
        subjects_set = set([get_file_info(subject_listing)['subject_id'] for subject_listing in subjects])
        self.images = []
        self.labels = []
        self.groups = []
        for extension in config.file_extensions:
            images_by_extension = sorted(base_dir.glob(f'**/*.{extension}'))
            for image in images_by_extension:
                image_info = get_file_info(image.name)
                subject_id = image_info['subject_id']
                
                if subject_id in subjects_set:
                    subject_images[subject_id].append(image)
                    self.images.append(image)
                    self.labels.append(image_info['label'])
                    self.groups.append(image_info['group'])
    
    def __len__(self):
        return len(self.images)    
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image(str(image_path))
        string_label = self.labels[index]
        label = 1. if string_label == 'A' else 0.
        metadata = {'image_path': str(image_path)}
        
        return dict(data=image, label=torch.tensor([label]), metadata=metadata)

    def get_collate_fn(self):
        return collate_face_dataset_view



def collate_face_dataset_view(batch):
    collated_dict = defaultdict(list)
    for record in batch:
        for k,v in record.items():
            collated_dict[k].append(v)
    collated_dict['data'] = torch.stack(collated_dict['data'])
    collated_dict['label'] = torch.stack(collated_dict['label'])
    return collated_dict


class EpochInformedSampler(Sampler):
    def __init__(self, dataset, shuffle=False, rng=None):
        if rng is None:
            rng = random.Random()
        self.rng = rng
        self.shuffle = shuffle
        self.dataset = dataset
        self.n = len(dataset)
        self.epoch = 0
        self.indices = np.arange(self.n)
    
    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        epoch = self.epoch
        for i in self.indices:
            yield (i, epoch)
        self.epoch += 1
        

class CachedFaceDatasetView(FaceDatasetView):
    def __init__(self, config: FaceDatasetConfig, base_dir: Path, subjects):
        self.name = config.name
        self.base_dir = base_dir
        
        self.data =  np.load(self.base_dir / 'data.npy', mmap_mode='r')
        self.max_epochs, self.samples, self.n_channels, self.height, self.width = self.data.shape
        with open(self.base_dir / 'files.txt') as fp:
            self.files = [Path(line) for line in fp]
        
        # Make a map to go from indices in this dataset given by *subjects* to the indices in the full cache file
        subjects_set = set([get_file_info(subject_listing)['subject_id'] for subject_listing in subjects])
        subjects_items = []
        self.labels = []
        self.groups = []
        
        for i, p in enumerate(self.files):
            image_info = get_file_info(p.name)
            subject_id = image_info['subject_id']
            
            if subject_id in subjects_set:
                subjects_items.append(i)
                self.labels.append(image_info['label'])
                self.groups.append(image_info['group'])
        
        self.index_map = np.array(subjects_items)
        
        self.epoch = 0
        self.seen_epoch_indices = set()
            
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, dataset_index):
        # We determine whether we are in a new epoch by looking 
        # whether this index has been used in the current epoch. If it has we assume we're in a new epoch
        if dataset_index in self.seen_epoch_indices:
            self.seen_epoch_indices = set()
            self.epoch += 1
            self.epoch %= self.max_epochs
        self.seen_epoch_indices.add(dataset_index)
        
        cache_file_index = self.index_map[dataset_index]
        data_item = self.data[self.epoch, cache_file_index]
        data = torch.tensor(data_item[:])
        label = self.labels[dataset_index]
        label = 1. if label == 'A' else 0.
        image_path = self.files[cache_file_index]
        metadata = {'image_path': str(image_path)}
        return dict(data=data, label=torch.tensor([label]), metadata=metadata)
    
    
    @classmethod
    def make_cache_dir(cls, cache_directory: Path, dataset_shape, dataloader: DataLoader, max_epochs: int):
        from tqdm import tqdm, trange
        cache_directory.mkdir(exist_ok=True, parents=True)
        data_array_path = cache_directory / 'data.npy'
        data_array = None
        files = []
        save_files = True
        for epoch in trange(max_epochs, desc="Epoch"):
            start = 0
            for i, batch in enumerate(tqdm(dataloader, desc="Batch")):
                data_item = batch['data'].cpu().numpy()
                batch_size, n_channels, height, width = data_item.shape
                if save_files:
                    batch_files = [record['image_path'] for record in batch['metadata']]
                    files.extend(batch_files)
                if data_array is None:
                    shape = (max_epochs, dataset_shape, n_channels, height, width)
                    data_array = open_memmap(data_array_path, mode='w+', shape=shape, dtype=data_item.dtype)
                end = start + batch_size
                data_array[epoch, start:end] = data_item
                start = end
            if save_files:
                with open(cache_directory / 'files.txt', 'w') as fp:
                    for file in files:
                        fp.write(f"{file}\n")
                save_files = False
                
                
                
                
                
                
                
            #     epoch_shapes.append(tuple(data_item.shape))
            #     data_item_path = epoch_dir / f"{i}.npy"
            #     epoch_files.append(data_item_path)
            #     np.save(data_item_path, data_item.cpu().numpy())
            # batch_sizes, channel_sizes, heights, widths = zip(*epoch_shapes)
            # epoch_size = sum(batch_sizes)
            # assert len(set(channel_sizes)) == 1, "Differing channel sizes in epoch"
            # assert len(set(heights)) == 1, "Differing heights in epoch"
            # assert len(set(widths)) == 1, "Differing widths in epoch"
            # n_channels = channel_sizes[0]
            # height = heights[0]
            # width = widths[0]
            # epoch_array_shape = (epoch_size, n_channels, height, width)
            
            # epoch_array_path = cache_directory / f"{epoch}.npy"
            
            # start = 0
            # for data_item_path in epoch_files:
            #     data_item = np.load(data_item_path)
            #     end = start + len(data_item)
            #     epoch_array[start:end] = data_item
            
                
                
        


class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms=None) -> None:
        super().__init__()
        self.images = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image(str(image_path))
        if self.transforms is not None:
            image = self.transforms(image)
        file_info = get_file_info(image_path.name)
        label = 1. if file_info['label'] == 'A' else 0.
        return image, torch.tensor([label]), image_path #, desired_yaw, tag

def collate_image_dataset(batch):
    images, labels, image_paths = zip(*batch)
    images_t = torch.stack(images)
    labels_t = torch.stack(labels)
    return images_t, labels_t, image_paths
    
    
class FaceDataset(AbstractDataset):
    def __init__(self, dataset_config: FaceDatasetConfig, *args, **kwargs) -> None:
        self.config = dataset_config
        # We want to determine what valid subjects there are. This should be in a list called "subjects.txt" with one code per line
        root_dir = Path(self.config.root_directory)
        subjects_file = root_dir / self.config.subject_index_file
        if subjects_file.exists():
            with open(subjects_file) as fp:
                self.subjects = [line.strip() for line in fp]
        else:
            # try to determine subjects from existing files
            training_subdir = root_dir / self.config.training_subdir
            test_subdir = root_dir / self.config.test_subdir
            train_subjects = set()
            for d in training_subdir.iterdir():
                try:
                    subject_id = get_file_info(d.name)['subject_id']
                    train_subjects.add(subject_id)
                except RuntimeError:
                    continue
            
            test_subjects = set()
            for d in test_subdir.iterdir():
                try:
                    subject_id = get_file_info(d.name)['subject_id']
                    test_subjects.add(subject_id)
                except RuntimeError:
                    continue
            
            self.subjects = sorted(train_subjects & test_subjects)
        super().__init__(*args, **kwargs)

    def get_index(self) -> List[int]:
        index = list(range(len(self.subjects)))
        return index
    
    def get_labels(self):
        infos = [get_file_info(self.subjects[i]) for i in self.indices]
        labels = [info['label'] for info in infos]
        return labels
    
    def get_groups(self):
        infos = [get_file_info(self.subjects[i]) for i in self.indices]
        groups = [info['group'] for info in infos]
        return groups

    def make_subset(self, indices=None):
        return FaceDataset(self.config, indices=indices)
    
    def default_split(self) -> "AbstractDataset":
        """
        Return the default trainval/test split of this dataset if there is one"""
        raise NotImplementedError(f'default_split() has not been implemented for {type(self)}')

    def instantiate_dataset(self, view_directory) -> DatasetView:
        subject_subset = [self.subjects[i] for i in self.indices]
        cache_path = view_directory / 'data.npy'
        if cache_path.exists():
            dataset_view = CachedFaceDatasetView(self.config, view_directory, subject_subset)
        else:
            dataset_view = FaceDatasetView(self.config, view_directory, subject_subset)
        return dataset_view

    def instantiate_training_dataset(self) -> DatasetView:
        view_directory = Path(self.config.root_directory) / self.config.training_subdir
        return self.instantiate_dataset(view_directory)
    
    def instantiate_test_dataset(self) -> DatasetView: 
        view_directory = Path(self.config.root_directory) / self.config.test_subdir
        return self.instantiate_dataset(view_directory)
        

class EncryptedFaceDataset(FaceDataset):
    def __init__(self, *args, key=None, **kwargs):
        super().__init__(*args, **kwargs)
        if key is None:
            self.passphrase = input("Please enter key")
            with open(Path(self.config.root_directory) / 'salt.dat', 'rb') as fp:
                self.salt = fp.read()
            key = derive_key(self.salt, self.passphrase)
        self.key = key
        
    def make_subset(self, indices=None):
        return EncryptedFaceDataset(self.config, indices=indices, key=self.key)
    
    def instantiate_dataset(self, view_directory) -> DatasetView:
        subject_subset = [self.subjects[i] for i in self.indices]
        cache_path = view_directory / 'data.npy'
        if cache_path.exists():
            dataset_view = EncryptedCachedFaceDatasetView(self.config, view_directory, subject_subset, self.key)
        else:
            dataset_view = EncryptedFaceDatasetView(self.config, view_directory, subject_subset, self.key)
        return dataset_view
    

        
class EncryptedFaceDatasetView(FaceDatasetView):
    def __init__(self, config: FaceDatasetConfig, base_dir: Path, subjects, key: bytes):
        super().__init__(config=config, base_dir=base_dir, subjects=subjects)
        self.key = key
        self.f = Fernet(key)
    
    def __len__(self):
        return len(self.images)    
    
    def __getitem__(self, index):
        image_path = self.images[index]
        with open(image_path, 'rb') as fp:
            encrypted_bytes = fp.read()
        decrypted_bytes = self.f.decrypt(encrypted_bytes)
        image_bytes = np.frombuffer(decrypted_bytes, dtype=np.uint8)
        image = decode_image(torch.tensor(image_bytes))
        string_label = self.labels[index]
        label = 1. if string_label == 'A' else 0.
        metadata = {'image_path': str(image_path)}
        
        return dict(data=image, label=torch.tensor([label]), metadata=metadata)

    def get_collate_fn(self):
        return collate_face_dataset_view
    

def encrypt_file(file_path, output_path, key):
    with open(file_path, 'rb') as fp:
        bytes = fp.read()
    
    f = Fernet(key)
    encrypted_bytes = f.encrypt(bytes)
    with open(output_path, 'wb') as fp:
        fp.write(encrypted_bytes)
        
def decrypt_file(file_path, key):
    with open(file_path, 'rb') as fp:
        encrypted_bytes = fp.read()
    f = Fernet(key)
    decrypted_bytes = f.decrypt(encrypted_bytes)
    return decrypted_bytes


def derive_key(salt, password):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    b_password = bytes(password, encoding='utf8')
    key = base64.urlsafe_b64encode(kdf.derive(b_password))
    return key
    

