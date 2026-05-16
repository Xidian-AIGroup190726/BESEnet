import logging
import random
from os import listdir
from os.path import splitext
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from config.hyperparameters import experiment_parameters


class DirectionFieldLabelEncoder:
    @classmethod
    def encode(cls, label):
        assert label.ndim == 2, "label must be a 2D array"

        category_mask = label.copy()
        category_mask[category_mask == 255] = 1
        category_mask += 1
        category_mask = category_mask.astype(np.float32)
        categories = np.unique(category_mask)
        if 0 in categories:
            raise RuntimeError("invalid category")
        category_mask = cv2.copyMakeBorder(category_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        height, width = label.shape
        direction_field = np.zeros((2, height + 2, width + 2), dtype=np.float32)
        weight_matrix = np.zeros((height + 2, width + 2), dtype=np.float32)

        for category in categories:
            binary_region = (category_mask == category).astype(np.uint8)
            weight_matrix[binary_region > 0] = 1.0 / np.sqrt(binary_region.sum())
            _, nearest_labels = cv2.distanceTransformWithLabels(
                binary_region,
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
                labelType=cv2.DIST_LABEL_PIXEL,
            )
            nearest_index = np.copy(nearest_labels)
            nearest_index[binary_region > 0] = 0
            nearest_coordinates = np.argwhere(nearest_index > 0)[nearest_labels - 1, :]
            nearest_pixel = np.zeros((2, height + 2, width + 2))
            nearest_pixel[0, :, :] = nearest_coordinates[:, :, 0]
            nearest_pixel[1, :, :] = nearest_coordinates[:, :, 1]
            spatial_grid = np.indices(binary_region.shape).astype(float)
            displacement = spatial_grid - nearest_pixel
            direction_field[:, binary_region > 0] = displacement[:, binary_region > 0]

        return direction_field[:, 1:-1, 1:-1], weight_matrix[1:-1, 1:-1]


class ChangeDetectionDataset(Dataset):
    def __init__(
            self,
            t1_images_dir: str,
            t2_images_dir: str,
            labels_dir: str,
            t1_seg_dir: str,
            t2_seg_dir: str,
            all_seg_dir: str,
            train: bool,
            t1_mean: list,
            t1_std: list,
            t2_mean: list,
            t2_std: list):
        self.t1_images_dir = Path(t1_images_dir)
        self.t2_images_dir = Path(t2_images_dir)
        self.labels_dir = Path(labels_dir)
        self.t1_seg_dir = Path(t1_seg_dir)
        self.t2_seg_dir = Path(t2_seg_dir)
        self.all_seg_dir = Path(all_seg_dir)
        self.train = train

        self.t1_ids = [splitext(file)[0] for file in listdir(t1_images_dir) if not file.startswith(".")]
        self.t2_ids = [splitext(file)[0] for file in listdir(t2_images_dir) if not file.startswith(".")]
        self.t1_ids.sort()
        self.t2_ids.sort()

        if not self.t1_ids:
            raise RuntimeError(f"No input file found in {t1_images_dir}, make sure you put your images there")
        if not self.t2_ids:
            raise RuntimeError(f"No input file found in {t2_images_dir}, make sure you put your images there")
        assert len(self.t1_ids) == len(self.t2_ids), "number of t1 images is not equivalent to number of t2 images"
        logging.info(f"Creating dataset with {len(self.t1_ids)} examples")

        self.joint_training_transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.5,
                ),
                A.Transpose(p=0.5),
                A.Rotate(45, p=0.3),
                A.ShiftScaleRotate(p=0.3),
            ],
            additional_targets={
                "image1": "image",
                "mask_t1_seg": "mask",
                "mask_t2_seg": "mask",
                "mask_all_seg": "mask",
            },
        )
        self.image_training_transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussNoise(p=1),
                        A.HueSaturationValue(p=1),
                        A.RandomBrightnessContrast(p=1),
                        A.RandomGamma(p=1),
                        A.Emboss(p=1),
                        A.MotionBlur(p=1),
                    ],
                    p=experiment_parameters.noise_p,
                )
            ],
            additional_targets={"image1": "image"},
        )
        self.t1_normalizer = A.Compose([A.Normalize(mean=t1_mean, std=t1_std)])
        self.t2_normalizer = A.Compose([A.Normalize(mean=t2_mean, std=t2_std)])
        self.tensor_transform = A.Compose(
            [ToTensorV2()],
            additional_targets={
                "image1": "image",
                "mask_t1_seg": "mask",
                "mask_t2_seg": "mask",
                "mask_all_seg": "mask",
            },
        )
        self.weight_tensor_transform = A.Compose(
            [ToTensorV2()],
            additional_targets={
                "weight_t1": "mask",
                "weight_t2": "mask",
                "weight_all": "mask",
            },
        )

    def __len__(self):
        return len(self.t1_ids)

    @classmethod
    def binarize_label(cls, label):
        label[label != 0] = 1
        return label

    @classmethod
    def load_image(cls, filename):
        return np.array(Image.open(filename))

    def __getitem__(self, index):
        t1_name = self.t1_ids[index]
        t2_name = self.t2_ids[index]
        assert t1_name == t2_name, f"t1 name{t1_name} not equal to t2 name{t2_name}"

        t1_image_file = list(self.t1_images_dir.glob(t1_name + ".*"))
        t2_image_file = list(self.t2_images_dir.glob(t2_name + ".*"))
        label_file = list(self.labels_dir.glob(t1_name + ".*"))
        t1_seg_file = list(self.t1_seg_dir.glob(t1_name + ".*"))
        t2_seg_file = list(self.t2_seg_dir.glob(t2_name + ".*"))

        assert len(label_file) == 1, f"Either no label or multiple labels found for the ID {t1_name}: {label_file}"
        assert len(t1_image_file) == 1, f"Either no image or multiple images found for the ID {t1_name}: {t1_image_file}"

        t1_image = self.load_image(t1_image_file[0])
        t2_image = self.load_image(t2_image_file[0])
        change_label = self.binarize_label(self.load_image(label_file[0]))

        t1_seg_label = self.binarize_label(self.load_image(t1_seg_file[0]))
        t1_seg_label = np.where(change_label == t1_seg_label, 1, 0)
        t2_seg_label = self.binarize_label(self.load_image(t2_seg_file[0]))
        t2_seg_label = np.where(change_label == t2_seg_label, 1, 0)
        all_seg_label = self.binarize_label(self.load_image(label_file[0]))

        sample = {
            "image": t1_image,
            "image1": t2_image,
            "mask": change_label,
            "mask_t1_seg": t1_seg_label,
            "mask_t2_seg": t2_seg_label,
            "mask_all_seg": all_seg_label,
        }

        if self.train:
            sample = self.joint_training_transforms(**sample)
            t1_image = sample["image"]
            t2_image = sample["image1"]
            change_label = sample["mask"]
            t1_seg_label = sample["mask_t1_seg"]
            t2_seg_label = sample["mask_t2_seg"]
            all_seg_label = sample["mask_all_seg"]
            sample = self.image_training_transforms(image=t1_image, image1=t2_image)
            t1_image = sample["image"]
            t2_image = sample["image1"]

        t1_image = self.t1_normalizer(image=t1_image)["image"]
        t2_image = self.t2_normalizer(image=t2_image)["image"]

        if self.train and random.choice([0, 1]):
            t1_image, t2_image = t2_image, t1_image
            t1_seg_label, t2_seg_label = t2_seg_label, t1_seg_label

        t1_direction, t1_weight = DirectionFieldLabelEncoder.encode(t1_seg_label)
        t2_direction, t2_weight = DirectionFieldLabelEncoder.encode(t2_seg_label)
        all_direction, all_weight = DirectionFieldLabelEncoder.encode(all_seg_label)

        tensor_sample = self.tensor_transform(
            image=t1_image,
            image1=t2_image,
            mask=change_label,
            mask_t1_seg=t1_direction.transpose(1, 2, 0),
            mask_t2_seg=t2_direction.transpose(1, 2, 0),
            mask_all_seg=all_direction.transpose(1, 2, 0),
        )
        weight_sample = self.weight_tensor_transform(
            image=t1_image,
            weight_t1=t1_weight,
            weight_t2=t2_weight,
            weight_all=all_weight,
        )

        t1_tensor = tensor_sample["image"].contiguous()
        t2_tensor = tensor_sample["image1"].contiguous()
        label_tensor = tensor_sample["mask"].contiguous()
        t1_direction_tensor = tensor_sample["mask_t1_seg"].contiguous().permute(2, 0, 1)
        t2_direction_tensor = tensor_sample["mask_t2_seg"].contiguous().permute(2, 0, 1)
        all_direction_tensor = tensor_sample["mask_all_seg"].contiguous().permute(2, 0, 1)
        t1_weight_tensor = weight_sample["weight_t1"].contiguous()
        t2_weight_tensor = weight_sample["weight_t2"].contiguous()
        all_weight_tensor = weight_sample["weight_all"].contiguous()

        return (
            t1_tensor,
            t2_tensor,
            label_tensor,
            t1_direction_tensor,
            t2_direction_tensor,
            all_direction_tensor,
            t1_weight_tensor,
            t2_weight_tensor,
            all_weight_tensor,
            t1_name,
        )
