from collections import defaultdict, deque
import csv
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field
import re
from typing import List, Tuple, Union, Literal, Optional
import tempfile
import shutil

from tqdm import trange, tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchvision.io import read_image, write_png
from torchvision.transforms.v2.functional import resize, to_pil_image
import pandas as pd

import PIL

import matplotlib.patches as patches



#celebmask_categories = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
#                        'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

@dataclass
class PreprocessingConfig:
    yaws: List[Union[int, float]]
    selected_files: List[Union[str, Path]] = field(default_factory=list)
    face_crop_ratio: float = 1.2
    target_width: Optional[int] = None #512
    target_height: Optional[int] = None #512
    mask_labels: List[Literal["cloth", "hat", "background"]] = field(default_factory=lambda: ["cloth"])
    keep_labels: List[str] = field(default_factory=lambda: ["skin", "hair", "l_brow", "r_brow", 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                        'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', "background"])
    discretization_threshold: float = 0.5
    mask_margin: float = 0
    #device: str = 'cuda'
    device: str = 'cpu'
    fix_ratio: bool = False
    batch_size: int = 32
    
 
class VideoDataset(IterableDataset):
    def __init__(self, video_path: Path, transforms=None) -> None:
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        self.cap.setExceptionMode(True)
    
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap.release()
        self.transforms = transforms
    

    def __len__(self):
        return self.n_frames


    def __iter__(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            _, img = self.cap.read()
            if img is None:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            img_t = torch.tensor(img)
            if self.transforms:
                img_t = self.transforms(img_t)
            yield img_t
        self.cap.release()


def preprocess_dataset(dataset_root: Path, config: PreprocessingConfig):
    video_files = sorted((dataset_root / 'raw').glob('**/*.mp4'))
    processed_root = dataset_root / 'processed'
    #cropped_video_files = crop_faces(video_files, processed_root, config)
    #cropped_video_files = show_bbox_faces(video_files, processed_root, config)
    estimated_yaw_files = estimate_yaws(video_files, processed_root, config)
    yaw_directories = select_yaws(estimated_yaw_files, processed_root, config)
    masked_images = mask_images(yaw_directories, processed_root, config)
    #concatenated_images = concatenate_images(masked_images, processed_root, config)


def select_yaws(yaw_files: List[Tuple[Path, Path]], processed_root: Path, config: PreprocessingConfig):
    yaw_shorthand = make_yaw_shorthand(config.yaws)
    yaw_root_directory = processed_root / yaw_shorthand
    yaw_root_directory.mkdir(parents=True, exist_ok=True)

    
    to_process = []
    selected_yaw_directories = []

    for yaw_file, video_file in yaw_files:
        video_name = video_file.name
        output_directory = yaw_root_directory / video_name
        all_exists = True
        yaw_file_targets = []
        for desired_yaw in config.yaws:
            pattern = f"{video_name}_{desired_yaw}*"
            existing_yaw_frames = list(output_directory.glob(pattern))
            if not existing_yaw_frames:
                output_file_pattern =  output_directory / f"{video_name}_{desired_yaw}"
                yaw_file_targets.append((desired_yaw, output_file_pattern))
                all_exists = False
        if all_exists:
            selected_yaw_directories.append(output_directory)
        else:
            to_process.append((yaw_file, video_file, yaw_file_targets, yaw_root_directory, config))

    with multiprocessing.Pool() as pool:
        for selected_yaw_directory in tqdm(pool.imap_unordered(select_from_file_yaws_pool_worker, to_process), desc="Selecting yaws", total=len(to_process)):
            selected_yaw_directories.append(selected_yaw_directory)
    return yaw_root_directory, selected_yaw_directories


def select_from_file_yaws_pool_worker(work_package):
    yaw_file, video_file, yaw_file_targets, yaw_root_directory, config = work_package
    selected_yaw_directory = select_from_file_yaws(yaw_file, video_file, yaw_root_directory, yaw_file_targets, config)
    return selected_yaw_directory


def make_yaw_shorthand(yaws):
    min_yaw = min(yaws)
    max_yaw = max(yaws)
    n_yaws = len(yaws)
    shorthand = f"yaws_{min_yaw}_{max_yaw}_{n_yaws}"
    return shorthand
    

def estimate_yaws(video_files: List[Path], processed_root: Path, config: PreprocessingConfig):
    output_dir = processed_root / f"yaws"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaw_files = []
    to_process_files = []
    for video_file in video_files:
        video_name = video_file.with_suffix("").name
        yaw_output_file = output_dir / f"{video_name}.csv"
        if not yaw_output_file.exists():
            to_process_files.append(video_file)
        else:
            output_yaw_files.append((yaw_output_file, video_file))
    
    if to_process_files:
        from acroface.models.sixdrepnet import YawPredictor
        model = YawPredictor(config.device)

        for video_file in tqdm(to_process_files, desc="Videos to process"):
            video_name = video_file.with_suffix("").name
            yaw_output_file = output_dir / f"{video_name}.csv"
            find_yaws(video_file, model, yaw_output_file, config)
            output_yaw_files.append((yaw_output_file, video_file))
    return output_yaw_files

def find_yaws(video_path: Path, model, yaw_csv_output: Path, config: PreprocessingConfig):
    cap = cv2.VideoCapture(str(video_path))
    cap.setExceptionMode(True)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pbar = trange(n_frames, desc="Frames", leave=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # First find the area of interest for the whole movie
    yaws = []
    frames = []

    batch = []
    batch_size = config.batch_size
    frame_index = 0
    while True:
        try:
            _, img = cap.read()
        except:
            break
        if img is None:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        batch.append((frame_index, img))
        if len(batch) >= batch_size:
            frame_indices, images = zip(*batch)
            arr_images = np.array(images)
            predicted_yaws = model.predict_yaws(arr_images)
            yaws.extend(predicted_yaws)
            frames.extend(frame_indices)
            batch.clear()
        pbar.update()
        frame_index += 1
    
    if batch:
        frame_indices, images = zip(*batch)
        arr_images = np.array(images)
        predicted_yaws = model.predict_yaws(arr_images)
        yaws.extend(predicted_yaws)
        frames.extend(frame_indices)
    
    cap.release()

    with open(yaw_csv_output, 'w') as fp:
        csv_writer = csv.DictWriter(fp, fieldnames=["frame", "yaw"])
        csv_writer.writeheader()
        for frame, yaw in zip(frames, yaws):
            csv_writer.writerow({'frame': frame, 'yaw': yaw.item()})
    

def select_from_file_yaws(yaw_file: Path, video_file: Path, yaw_directory: Path, yaw_file_targets, config: PreprocessingConfig):
    video_name = video_file.name
    output_directory = yaw_directory / video_name
    output_directory.mkdir(exist_ok=True, parents=True)

    yaws_df = pd.read_csv(yaw_file)
    frames_array = yaws_df['frame']
    yaws_array = yaws_df['yaw'].values 
    desired_yaws = np.array([yaw for yaw, target_file_pattern in yaw_file_targets])

    # While searchsorted is useful for very large arrays, it doesn't locate the closest frame. 
    #yaws_sort_index = np.argsort(yaws_array)
    #sorted_yaws = yaws_array[yaws_sort_index]
    #closest_yaws = np.searchsorted(sorted_yaws, desired_yaws)
    
    # We broadcast the subtraction to create on difference per desired yaw
    yaw_diffs = yaws_array[:, None] - desired_yaws[None, :]
    # yaw_diffs is shape (n_frames, n_desired_yaws). 
    # We take the argmin of the absolute values to find the correct indices of the closest yaws
    yaw_indices = np.argmin(np.abs(yaw_diffs), axis=0)
    yaw_selector = deque(sorted([(frames_array[yaw_index], yaw, output_file_pattern, yaws_array[yaw_index]) for yaw_index, (yaw, output_file_pattern) in zip(yaw_indices, yaw_file_targets)]))


    cap = cv2.VideoCapture(str(video_file))
    cap.setExceptionMode(False)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pbar = trange(n_frames, desc="Frames", leave=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    #target_frame, target_yaw, target_file_pattern, image_yaw = yaw_selector.popleft()
    frame_idx = 0
    _, img = cap.read()
    if img is None:
        raise RuntimeError(f"Video {video_file} seems to be empty")
    current_img = img
    for target_frame, target_yaw, target_file_pattern, image_yaw in yaw_selector:

        while frame_idx < target_frame or img is None:
            _, img = cap.read()
            frame_idx += 1
            if img is None:
                break
            current_img = img
        
        target_directory = target_file_pattern.parent
        target_file_name = target_file_pattern.name
        output_file = target_directory / f"{target_file_name}_{image_yaw:.2f}.png"
        cv2.imwrite(str(output_file), current_img)
        pbar.update()        

    cap.release()

    return output_directory



def crop_faces(video_files: List[Path], processed_root: Path, config: PreprocessingConfig):
    output_dir = processed_root / f"cropped_{config.face_crop_ratio:.1f}_{config.target_width}x{config.target_height}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_files = []
    to_process_files = []
    for video_file in video_files:
        output_video_path = output_dir / video_file.name
        if not output_video_path.exists():
            to_process_files.append(video_file)
        else:
            output_video_files.append(output_video_path)
    
    if to_process_files:
        # We essentially hide the dependency on batch_face until it's needed
        from batch_face import RetinaFace
        model = RetinaFace(0)
        for video_file in tqdm(to_process_files, desc="Videos to process"):
            output_video_path = output_dir / video_file.name
            #find_crop_area_from_mask(video_file, output_video_path, model, config)
            find_crop_area_from_mask(video_file, output_video_path, config)
            output_video_files.append(output_video_path)
    return output_video_files
        

def find_crop_area_from_mask(video_path: Path, output_video_path: Path, config: PreprocessingConfig):
    import torch
    from acroface.models import bisenet
    import matplotlib.pyplot as plt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = bisenet.load_model(device=device)
    model = model.eval()
    transforms = bisenet.get_eval_transforms()
    
    video_dataset = VideoDataset(video_path, transforms=transforms)
    video_dataloader = DataLoader(video_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, prefetch_factor=None)
    width = video_dataset.width
    height = video_dataset.height
    

    with torch.inference_mode():
        n_classes = len(bisenet.celebmask_categories)
        
        mask_indices = bisenet.get_label_indices(config.mask_labels)
        n_samples = 0
        summed_probabilities = None
        for batch in tqdm(video_dataloader, desc="Video batches"):
            predicted_mask = bisenet.predict_mask(batch, model, device, video_dataset.width, video_dataset.height)
            if summed_probabilities is None:
                summed_probabilities = predicted_mask.sum(dim=0).cpu().numpy()
            else:
                summed_probabilities += predicted_mask.sum(dim=0).cpu().numpy()
            n_samples += len(predicted_mask)

    mean_probabilties = summed_probabilities / n_samples
    discretized_mask_predictions = mean_probabilties[mask_indices] > config.discretization_threshold
    # fig, axes = plt.subplots(1, len(mask_indices))
    # for ax, mask_index in zip(axes.flatten(), mask_indices):
    #     mask = mean_probabilties[mask_index] > config.discretization_threshold
    #     ax.imshow(mask.astype(float))
    # plt.show()

    composite_discretization = discretized_mask_predictions.any(axis=0)
    all_masked_rows = composite_discretization.all(axis=1)
    all_masked_cols = composite_discretization.all(axis=0)
    non_masked_rows = np.where(np.logical_not(all_masked_rows))
    non_masked_cols = np.where(np.logical_not(all_masked_cols))

    min_x = np.min(non_masked_cols)
    max_x = np.max(non_masked_cols)
    min_y = np.min(non_masked_rows)
    max_y = np.max(non_masked_rows)
        
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    bbox_center_x = min_x + bbox_width/2
    bbox_center_y = min_y + bbox_height/2
    
    # We scale the bounding box and make sure it fits within the image
    scaled_bbox_width = bbox_width*config.face_crop_ratio
    scaled_bbox_height = bbox_height*config.face_crop_ratio

    bbox_width_difference = scaled_bbox_width - bbox_width
    bbox_height_difference = scaled_bbox_height - bbox_height
    min_x = max(0, min_x - bbox_width_difference/2)
    max_x = min(width, max_x + bbox_width_difference/2)
    min_y = max(0, min_y - bbox_height_difference/2)
    max_y = min(height, max_y + bbox_height_difference/2)

    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    bbox_center_x = min_x + bbox_width/2
    bbox_center_y = min_y + bbox_height/2

    if config.fix_ratio:
        bbox_ratio = bbox_height/bbox_width
        target_ratio = config.target_height/config.target_width

        # If the ratio is greater than the target ratio, the
        # bbox height is larger and should be decreased in size
        if bbox_ratio > target_ratio:
            bbox_height = bbox_width*target_ratio
        # If the bbox ratio is smaller than the target, the 
        # bbox width should be decreased in size
        else:
            bbox_width = bbox_height/target_ratio

        # Recalculate the absolute coordinates of the 
        # bounding box with the adjusted bounding box
        min_x = int(round(bbox_center_x - bbox_width/2))
        max_x = int(round(bbox_center_x + bbox_width/2))
        min_y = int(round(bbox_center_y - bbox_height/2))
        max_y = int(round(bbox_center_y + bbox_height/2))
    
    min_x, min_y = int(min_x), int(min_y)
    max_x, max_y = int(max_x), int(max_y)
    bbox_width, bbox_height = int(bbox_width), int(bbox_height)

    # fig, ax = plt.subplots()
    # ax.imshow(composite_discretization.astype(float))
    
    # # Create a Rectangle patch
    # rect = patches.Rectangle((min_x, min_y), bbox_width, bbox_height, linewidth=1, edgecolor='r', facecolor='none')

    # # Add the patch to the Axes
    # ax.add_patch(rect)
    # plt.show()
        
    cap = cv2.VideoCapture(str(video_path))
    cap.setExceptionMode(True)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.setExceptionMode(False)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pbar = trange(n_frames, desc="Frames", leave=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Create the output video writer
    fourcc = cv2.VideoWriter.fourcc('m','p','4','v')

    target_width, target_height = config.target_width, config.target_height
    if target_width is None:
        target_width = bbox_width
    if target_height is None:
        target_height = bbox_height

    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (target_width, target_height))

    pbar = trange(n_frames, desc="Frames", leave=False)
    while True:
        _, img = cap.read()
        if img is None:
            break
        cropped = img[min_y:max_y, min_x:max_x]
        resized = cv2.resize(cropped, (target_width, target_height))
        out.write(resized)
        pbar.update()
    cap.release()
    out.release()


    # out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (config.target_width, config.target_height))

    # pbar = trange(n_frames, desc="Writing frames", leave=False)
    # while True:
    #     _, img = cap.read()
    #     if img is None:
    #         break
    #     cropped = img[min_y:max_y, min_x:max_x]
    #     resized = cv2.resize(cropped, (config.target_width, config.target_height))
    #     out.write(resized)
    #     pbar.update()
    # cap.release()
      

class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms=None) -> None:
        super().__init__()
        self.images = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        try:
            original_image = read_image(str(image_path))
        except RuntimeError as e:
            print(f"Error reading image {image_path}")
            raise e
        if self.transforms is not None:
            transformed_image = self.transforms(original_image)
        else:
            transformed_image = original_image
        return original_image, transformed_image, image_path


def collate_fn(samples):
    original_images, transformed_images, paths = zip(*samples)
    transformed_images = torch.stack(list(transformed_images))
    return list(original_images), transformed_images, list(paths)


def mask_images(yaw_directories: Tuple[Path, List[Path]], processed_root: Path, config):
    yaw_root_directory, selected_yaws_directories = yaw_directories
    mask_root_directory = processed_root / f"masked_{yaw_root_directory.name}"
    yaw_images = [image for selected_yaw_directory in sorted(selected_yaws_directories) for image in selected_yaw_directory.glob("*.png")]

    to_process_images = []
    for image_path in yaw_images:
        image_relative_path = image_path.relative_to(yaw_root_directory)
        masked_image_output_path = mask_root_directory / image_relative_path
        if not masked_image_output_path.exists():
            to_process_images.append(image_path)
    
    import torch
    from acroface.models import bisenet
    import matplotlib.pyplot as plt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = bisenet.load_model(device=device)
    model = model.eval()
    transforms = bisenet.get_eval_transforms()

    image_dataset = ImageDataset(to_process_images, transforms=transforms)
#    image_dataloader = DataLoader(image_dataset, batch_size=config.batch_size, num_workers=4, drop_last=False,collate_fn=collate_fn)
    image_dataloader = DataLoader(image_dataset, batch_size=config.batch_size, num_workers=0, drop_last=False,collate_fn=collate_fn)
    processed_images = []
    with torch.inference_mode():
        n_classes = len(bisenet.celebmask_categories)  
        
        mask_indices = bisenet.get_label_indices(config.mask_labels)
        #keep_indices = bisenet.get_label_indices(config.keep_labels)

        
        for batch in tqdm(image_dataloader, desc='Masking images'):
            original_images, transformed_images, paths = batch
            predicted_masks = bisenet.predict_mask(transformed_images, model, device)
            for original_image, predicted_mask, path in zip(original_images, predicted_masks, paths):
                n_channels, height, width = original_image.shape
                reshaped_predictions = resize(predicted_mask, (height, width))
                
                #This takes the max prediction, but requires it to be above a certain margin from the other predictions
                #reshaped_predictions[mask_indices] -= config.mask_margin
                #max_prediction = torch.argmax(reshaped_predictions, dim=0)
                # We compare each entry in max_prediction with the mask indices using broadcasting. This will result in a tensor of shape
                # height, width, len(mask_labels)
                #comparison_tensor = torch.tensor([[mask_indices]], device=max_prediction.device) == max_prediction.unsqueeze(2)
                #fill_mask = torch.any(comparison_tensor, dim=2).cpu()  # if any of the last dim elements are true the max prediction was in mask_labels
                
                # This dicretizes the prediction based on a fixed prediction probability threshold
                discretized_mask_predictions = reshaped_predictions[mask_indices] < config.discretization_threshold
                fill_mask = discretized_mask_predictions.any(dim=0, keepdims=True).cpu()
                
                # TODO: this is a hack to try to limit masking to the actual clothing (and not the nose bridge)
                #channels, height, width = original_image.shape
                #fill_mask[:int(height*0.75)] = False
                
                # This version flips it and uses a threshold on a "to keep" class
                #discretized_mask_predictions = reshaped_predictions[keep_indices] > config.discretization_threshold
                #keep_mask = discretized_mask_predictions.any(dim=0, keepdims=True).cpu()
                #fill_mask = torch.logical_not(keep_mask)
                
                # We will create the mask as the alpha channel of a PNG image. This means we will 
                # first create an image with the same size as the image but a single channel
                alpha_channel = fill_mask.to(dtype=torch.uint8)*255
                pil_alpha = to_pil_image(alpha_channel)
                
                pil_image = to_pil_image(original_image)
                pil_image.putalpha(pil_alpha)
                
                #plt.imshow(original_image.permute(1,2,0))
                #plt.show()
                image_relative_path = path.relative_to(yaw_root_directory)
                masked_image_output_path = mask_root_directory / image_relative_path
                masked_image_output_path.parent.mkdir(exist_ok=True, parents=True)
                pil_image.save(masked_image_output_path, format='PNG')
                processed_images.append(masked_image_output_path)

            # predicted_mask = bisenet.predict_mask(batch, model, device, video_dataset.width, video_dataset.height)
            # if summed_probabilities is None:
            #     summed_probabilities = predicted_mask.sum(dim=0).cpu().numpy()
            # else:
            #     summed_probabilities += predicted_mask.sum(dim=0).cpu().numpy()
            # n_samples += len(predicted_mask)
    return processed_images

         
def concatenate_images(images, processed_root: Path, config):
    yaw_shorthand = make_yaw_shorthand(config.yaws)
    output_directory = processed_root / f"concatenated_{yaw_shorthand}"
    video_yaw_images = defaultdict(list)
    for image in images:
        m = re.match(r".*_(\d*\.?\d*)_(-?\d*\.?\d*)\.png", image.name)
        if m is not None:
            target_yaw, actual_yaw = m.groups()
            video_yaw_images[image.parent].append((target_yaw, actual_yaw, image))
    for video_path, yaw_images in video_yaw_images.items():
        images = [cv2.imread(str(image)) for target_yaw, actual_yaw, image in sorted(yaw_images)]
        # OpenCV has the format (height, width, channels, we concatenate on axis=1, width)
        if images:
            concatenated_images = np.concatenate(images, axis=1)
            output_path = output_directory / (video_path.name + '_concatenated.jpg')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path),concatenated_images)
