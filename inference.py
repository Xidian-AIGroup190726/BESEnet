import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist
from skimage.segmentation import find_boundaries
from torch.utils.data import DataLoader
from torchmetrics import F1Score, JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from config.hyperparameters import experiment_parameters
from data.change_detection_dataset import ChangeDetectionDataset
from data.dataset_statistics import calculate_image_channel_statistics
from models.dpcd_network import ProfessionalChangeDetectionNetwork


def calculate_boundary_f1_score(prediction_mask, target_mask, tolerance=10):
    prediction_boundary = find_boundaries(prediction_mask.astype(bool), mode="outer").astype(np.uint8)
    target_boundary = find_boundaries(target_mask.astype(bool), mode="outer").astype(np.uint8)
    prediction_coordinates = np.argwhere(prediction_boundary)
    target_coordinates = np.argwhere(target_boundary)

    if len(prediction_coordinates) == 0 or len(target_coordinates) == 0:
        return 0.0

    distance_matrix = cdist(prediction_coordinates, target_coordinates)
    matched_prediction = np.any(distance_matrix <= tolerance, axis=1).sum()
    matched_target = np.any(distance_matrix <= tolerance, axis=0).sum()
    precision = matched_prediction / len(prediction_coordinates)
    recall = matched_target / len(target_coordinates)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_boundary_iou(prediction_mask, target_mask, boundary_width=10):
    prediction_boundary = find_boundaries(prediction_mask.astype(bool), mode="outer")
    target_boundary = find_boundaries(target_mask.astype(bool), mode="outer")
    prediction_dilation = binary_dilation(prediction_boundary, iterations=boundary_width)
    target_dilation = binary_dilation(target_boundary, iterations=boundary_width)
    intersection = np.logical_and(prediction_dilation, target_dilation).sum()
    union = np.logical_or(prediction_dilation, target_dilation).sum()
    return intersection / union if union > 0 else 0.0


def build_inference_dataset(dataset_name):
    t1_mean, t1_std = calculate_image_channel_statistics(images_dir=f"./{dataset_name}/train/t1/")
    t2_mean, t2_std = calculate_image_channel_statistics(images_dir=f"./{dataset_name}/train/t2/")
    normalization_config = {
        "t1_mean": t1_mean.tolist(),
        "t1_std": t1_std.tolist(),
        "t2_mean": t2_mean.tolist(),
        "t2_std": t2_std.tolist(),
    }
    return ChangeDetectionDataset(
        t1_images_dir=f"./{dataset_name}/test/t1/",
        t2_images_dir=f"./{dataset_name}/test/t2/",
        labels_dir=f"./{dataset_name}/test/change_label/",
        t1_seg_dir=f"./{dataset_name}/test/t1_label/",
        t2_seg_dir=f"./{dataset_name}/test/t2_label/",
        all_seg_dir=f"./{dataset_name}/test/all_label/",
        train=False,
        **normalization_config,
    )


def load_inference_network(device, load_checkpoint=True):
    assert experiment_parameters.load, "Loading model error, checkpoint experiment_parameters.load"
    network = ProfessionalChangeDetectionNetwork().to(device=device)
    checkpoint = torch.load(experiment_parameters.load, map_location=device)
    if load_checkpoint:
        network.load_state_dict(checkpoint["net"])
    else:
        network.load_state_dict(checkpoint)
    logging.info(f"Model loaded from {experiment_parameters.load}")
    network.eval()
    return network


def save_prediction_outputs(change_probability, label, sample_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_array = change_probability.cpu().numpy()
    binary_prediction = (prediction_array >= 0.5).astype(np.uint8)
    label_array = label.cpu().numpy().astype(np.uint8)

    mask_image = Image.fromarray((prediction_array * 255).astype(np.uint8), mode="L")
    mask_image.save(output_dir / f"{sample_name}_probability.png")

    color_map = {
        "FP": [255, 0, 0],
        "FN": [0, 0, 255],
        "TN": [0, 0, 0],
        "TP": [255, 255, 255],
    }
    diagnostic_image = np.zeros((prediction_array.shape[0], prediction_array.shape[1], 3), dtype=np.uint8)
    diagnostic_image[(binary_prediction == 1) & (label_array == 0)] = color_map["FP"]
    diagnostic_image[(binary_prediction == 0) & (label_array == 1)] = color_map["FN"]
    diagnostic_image[(binary_prediction == 0) & (label_array == 0)] = color_map["TN"]
    diagnostic_image[(binary_prediction == 1) & (label_array == 1)] = color_map["TP"]
    Image.fromarray(diagnostic_image, mode="RGB").save(output_dir / f"{sample_name}_diagnostic.png")

    return binary_prediction, label_array


def run_inference(dataset_name, load_checkpoint=True):
    test_dataset = build_inference_dataset(dataset_name)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=experiment_parameters.batch_size * experiment_parameters.inference_ratio,
        num_workers=4,
        prefetch_factor=5,
        persistent_workers=True,
    )

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    network = load_inference_network(device, load_checkpoint=load_checkpoint)
    torch.save(network.state_dict(), f"{dataset_name}_best_model.pth")

    metric_collection = MetricCollection({
        "iou": JaccardIndex(task="binary", num_classes=2).to(device=device),
        "precision": Precision(task="binary").to(device=device),
        "recall": Recall(task="binary").to(device=device),
        "f1score": F1Score(task="binary").to(device=device),
    })
    boundary_f1_scores = []
    boundary_ious = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            (
                t1_image,
                t2_image,
                labels,
                t1_direction_label,
                t2_direction_label,
                all_direction_label,
                t1_weight,
                t2_weight,
                all_weight,
                names,
            ) = batch
            t1_image = t1_image.float().to(device)
            t2_image = t2_image.float().to(device)
            labels = labels.float().to(device)

            change_logits, _, _, _, _ = network(t1_image, t2_image)
            change_probability = torch.sigmoid(change_logits).squeeze(1)

            for sample_index in range(change_probability.size(0)):
                metric_collection.update(change_probability[sample_index], labels[sample_index])
                prediction_mask, target_mask = save_prediction_outputs(
                    change_probability[sample_index],
                    labels[sample_index],
                    names[sample_index],
                    "./test_result/",
                )
                boundary_f1_scores.append(calculate_boundary_f1_score(prediction_mask, target_mask, tolerance=10))
                boundary_ious.append(calculate_boundary_iou(prediction_mask, target_mask, boundary_width=10))

            del t1_image, t2_image, labels, t1_direction_label, t2_direction_label, all_direction_label
            del t1_weight, t2_weight, all_weight

        test_metrics = metric_collection.compute()
        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()

        if boundary_f1_scores:
            print(f"Boundary F1-score: {np.mean(boundary_f1_scores):.4f}")
            print(f"Boundary IoU: {np.mean(boundary_ious):.4f}")

    print("over")


if __name__ == "__main__":
    try:
        run_inference(dataset_name="whu", load_checkpoint=False)
    except KeyboardInterrupt:
        logging.info("Error")
        sys.exit(0)
