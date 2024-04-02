import torch
import os
import os.path as osp
import numpy as np
import imgaug.augmenters as iaa
from utils import get_logger
from typing import Iterable, Tuple, List
from PIL import Image
from itertools import combinations, permutations
from tqdm import tqdm

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root:str, 
        n_classes: int,
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        is_train: bool = True,
    ):
        super().__init__()
        assert osp.exists(osp.join(root, 'images')), f'Path {root}/images does not exist'
        assert osp.exists(osp.join(root, 'labels')), f'Path {root}/labels does not exist'
        
        self.root = root
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.resize = resize
        self.is_train = is_train
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.logger = get_logger(__class__.__name__, 0) # TODO: Bug, every process will log
        
        self.paths = []
        for path in os.listdir(osp.join(self.root, 'images')):
            img_path = osp.join(self.root, 'images', path)
            label_path = osp.join(self.root, 'labels', path)
            if not osp.exists(label_path):
                self.logger.warn(f'Skipping label path {label_path} as it does not exist')
                continue
            self.paths.append((img_path, label_path))
        
        self._preload_dataset()
        self._generate_pairs()
        self._init_augmentation()
        self._filter_pairs()

    def _preload_dataset(self):
        self.images = []
        self.labels = []
        self.unique_classes = []
        for img_path, label_path in tqdm(self.paths, desc='Caching images and labels'):
            img = self._load_img(img_path)
            label = self._load_lbl(label_path)
            self.images.append(img)
            self.labels.append(label)
            self.unique_classes.append(set(np.unique(label)))

    def _generate_pairs(self):
        indices = np.arange(len(self.paths))
        if self.is_train:
            self.pairs = list(combinations(indices, 2))
        else:
            self.pairs = list(permutations(indices, 2))
    
    def _filter_pairs(self):
        self.same_class_pairs = []
        self.diff_class_pairs = []
        for pair in tqdm(self.pairs, desc='Filtering pairs'):
            len_intersect = self.unique_classes[pair[0]].intersection(self.unique_classes[pair[1]])
            len_union = self.unique_classes[pair[0]].union(self.unique_classes[pair[1]])
            if len_intersect == len_union:
                self.same_class_pairs.append(pair)
            else:
                self.diff_class_pairs.append(pair)
        np.random.shuffle(self.same_class_pairs)
        np.random.shuffle(self.diff_class_pairs)

    def _load_img(self, path):
        img = Image.open(path).convert('RGB')
        if self.resize is not None:
            img = img.resize(self.resize)
        img = np.array(img).astype(np.uint8)
        return img
    
    def _load_lbl(self, path):
        label = Image.open(path).convert('L')
        if self.resize is not None:
            label = label.resize(self.resize, Image.NEAREST)
        label = np.array(label).astype(np.uint8)
        return label

    def _load_img_lbl(self, img_path, label_path):
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        if self.resize is not None:
            img = img.resize(self.resize)
            label = label.resize(self.resize, Image.NEAREST)
        img = np.array(img).astype(np.uint8)
        label = np.array(label).astype(np.uint8)
        return img, label

    def _generate_color_palette(self):
        return np.random.randint(0, 256, (self.n_classes, 3))
    
    def _init_augmentation(self):
        if self.is_train:
            self.augment_all = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.CropToFixedSize(448, 448)
            ])
            self.augment_img = iaa.Sequential([ # does not change relative positions
                iaa.Multiply((0.9, 1.1), per_channel=0.5),
                iaa.GaussianBlur((0, 0.1))
            ])
        else:
            self.augment_all = iaa.Sequential([
                iaa.CropToFixedSize(448, 448)
            ])
            self.augment_img = iaa.Sequential([])
    
    def _augment(self, img: List[np.ndarray], label: List[np.ndarray], ori_label: List[np.ndarray]):
        aug_all = self.augment_all.to_deterministic()
        all = np.concatenate([img, label], axis=0)
        res = [aug_all.augment_image(x) for x in all]
        ori_label = [aug_all.augment_image(x) for x in ori_label]
        img, label = res[:len(img)], res[len(img):]
        img = self.augment_img.augment_images(img)
        return img, label, ori_label

    def _lbl_random_color(self, label: np.ndarray, color_palette: np.ndarray):
        result = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for i in range(self.n_classes):
            result[label == i] = color_palette[i]
        return result

    def _to_img_tensor(self, arr: np.ndarray):
        arr = arr / 255.0
        arr = arr - self.mean
        arr = arr / self.std
        res = torch.FloatTensor(arr)
        res = torch.einsum('hwc->chw', res)
        return res
    
    def _generate_mask(self, img_shape: Tuple[int, int], is_half: bool = False):
        # 1 means masked, 0 means not masked
        total_patch = (img_shape[0] // self.patch_size[0]) * (img_shape[1] // self.patch_size[1])
        if is_half:
            mask = torch.zeros(total_patch, dtype=torch.float32)
            mask[total_patch//2:] = 1
        else:
            total_ones = int(total_patch * self.mask_ratio)
            shuffle_idx = torch.randperm(total_patch)
            mask = torch.FloatTensor([0] * (total_patch - total_ones) + [1] * total_ones)[shuffle_idx]

        return mask

    def __getitem__(self, idx):
        if self.is_train:
            if idx < len(self.same_class_pairs):
                pair_idx1, pair_idx2 = self.same_class_pairs[idx]
            else:
                pair_idx1, pair_idx2 = self.diff_class_pairs[idx - len(self.same_class_pairs)]

            if np.random.rand() > 0.5: # swap pair
                pair_idx1, pair_idx2 = pair_idx2, pair_idx1
        else:
            pair_idx1, pair_idx2 = self.same_class_pairs[idx]

        img1, ori_label1 = self.images[pair_idx1], self.labels[pair_idx1]
        img2, ori_label2 = self.images[pair_idx2], self.labels[pair_idx2]

        color_palette = self._generate_color_palette()
        label1 = self._lbl_random_color(ori_label1, color_palette)
        label2 = self._lbl_random_color(ori_label2, color_palette)

        img, label, ori_label = self._augment([img1, img2], [label1, label2], [ori_label1, ori_label2])
        img = np.concatenate(img, axis=0)
        label = np.concatenate(label, axis=0)
        ori_label = np.concatenate(ori_label, axis=0)
        
        img = self._to_img_tensor(img)
        label = self._to_img_tensor(label)
        ori_label = torch.FloatTensor(ori_label)
        
        if not self.is_train:
            is_half = True
        else:
            is_half = idx < len(self.same_class_pairs)
        mask = self._generate_mask((img.shape[1], img.shape[2]), is_half)
        valid = torch.ones_like(label)
        seg_type = torch.zeros([1])
        color_palette = torch.FloatTensor(color_palette)
        return img, label, mask, valid, seg_type, ori_label, color_palette

    def __len__(self):
        if self.is_train:
            return len(self.same_class_pairs) + len(self.diff_class_pairs)
        return min(len(self.same_class_pairs), 1600)
