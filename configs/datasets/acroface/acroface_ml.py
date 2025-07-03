from acroface.dataset.core import DatasetConfig
from acroface.dataset.face import FaceDatasetConfig, FaceDataset


face_dataset_config = FaceDatasetConfig(name="ACROFace",
                                        root_directory='dataset/acroface_ml_dataset/', 
                                        training_subdir='training_dataset',
                                        test_subdir='test_dataset')


config = DatasetConfig(dataset_class=FaceDataset, dataset_args=(face_dataset_config,))