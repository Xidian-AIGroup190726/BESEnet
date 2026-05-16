import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch import optim
from torchmetrics import F1Score, JaccardIndex, MetricCollection, Precision, Recall

from config.hyperparameters import experiment_parameters
from data.change_detection_dataset import ChangeDetectionDataset
from data.dataset_statistics import calculate_image_channel_statistics
from data.prefetching import PrefetchDataLoader
from models.dpcd_network import ProfessionalChangeDetectionNetwork
from training.epoch_runner import execute_epoch
from training.objectives import calculate_multitask_change_detection_loss


def configure_reproducible_execution(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_configured_experiment():
    configure_reproducible_execution(experiment_parameters.random_seed)
    try:
        train_change_detection_network(dataset_name=experiment_parameters.dataset_name)
    except KeyboardInterrupt:
        logging.info("Interrupted")
        sys.exit(0)


def build_dataset_statistics(dataset_name):
    t1_mean, t1_std = calculate_image_channel_statistics(images_dir=f"./{dataset_name}/train/t1/")
    t2_mean, t2_std = calculate_image_channel_statistics(images_dir=f"./{dataset_name}/train/t2/")
    return {
        "t1_mean": t1_mean.tolist(),
        "t1_std": t1_std.tolist(),
        "t2_mean": t2_mean.tolist(),
        "t2_std": t2_std.tolist(),
    }


def build_change_detection_datasets(dataset_name, normalization_config):
    train_dataset = ChangeDetectionDataset(
        t1_images_dir=f"./{dataset_name}/train/t1/",
        t2_images_dir=f"./{dataset_name}/train/t2/",
        labels_dir=f"./{dataset_name}/train/change_label/",
        t1_seg_dir=f"./{dataset_name}/train/t1_label/",
        t2_seg_dir=f"./{dataset_name}/train/t2_label/",
        all_seg_dir=f"./{dataset_name}/train/all_label/",
        train=True,
        **normalization_config,
    )
    validation_dataset = ChangeDetectionDataset(
        t1_images_dir=f"./{dataset_name}/val/t1/",
        t2_images_dir=f"./{dataset_name}/val/t2/",
        labels_dir=f"./{dataset_name}/val/change_label/",
        t1_seg_dir=f"./{dataset_name}/val/t1_label/",
        t2_seg_dir=f"./{dataset_name}/val/t2_label/",
        all_seg_dir=f"./{dataset_name}/val/all_label/",
        train=False,
        **normalization_config,
    )
    return train_dataset, validation_dataset


def initialize_training_log(log_path, train_size, validation_size, device):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as logger:
        logger.write(f"""Starting training:
    Epochs:          {experiment_parameters.epochs}
    Batch size:      {experiment_parameters.batch_size}
    Learning rate:   {experiment_parameters.learning_rate}
    Training size:   {train_size}
    Validation size: {validation_size}
    Checkpoints:     {experiment_parameters.save_checkpoint}
    save best model: {experiment_parameters.save_best_model}
    Device:          {device.type}
    Mixed Precision: {experiment_parameters.amp}
    Started at:      {time.asctime(time.localtime(time.time()))}
    """)


def train_change_detection_network(dataset_name):
    normalization_config = build_dataset_statistics(dataset_name)
    train_dataset, validation_dataset = build_change_detection_datasets(dataset_name, normalization_config)

    loader_config = {
        "num_workers": 4,
        "prefetch_factor": 5,
        "persistent_workers": True,
        "pin_memory": True,
    }
    train_loader = PrefetchDataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        batch_size=experiment_parameters.batch_size,
        **loader_config,
    )
    validation_loader = PrefetchDataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=experiment_parameters.batch_size * experiment_parameters.inference_ratio,
        **loader_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    log_path = experiment_parameters.save_dir + experiment_parameters.log_file
    initialize_training_log(log_path, len(train_dataset), len(validation_dataset), device)

    network = ProfessionalChangeDetectionNetwork().to(device=device)
    optimizer = optim.AdamW(
        network.parameters(),
        lr=experiment_parameters.learning_rate,
        weight_decay=experiment_parameters.weight_decay,
    )
    warmup_learning_rates = np.arange(
        1e-7,
        experiment_parameters.learning_rate,
        (experiment_parameters.learning_rate - 1e-7) / experiment_parameters.warm_up_step,
    )
    gradient_scaler = torch.cuda.amp.GradScaler()

    if experiment_parameters.load:
        checkpoint = torch.load(experiment_parameters.load, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["net"])
        logging.info(f"Model loaded from {experiment_parameters.load}")
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = experiment_parameters.learning_rate
            optimizer.param_groups[0]["capturable"] = True

    total_step = 0
    learning_rate = experiment_parameters.learning_rate
    best_metrics = {"best_f1score": 0, "lowest loss": float("inf")}
    metric_collection = MetricCollection({
        "iou": JaccardIndex(task="binary", num_classes=2).to(device=device),
        "precision": Precision(task="binary").to(device=device),
        "recall": Recall(task="binary").to(device=device),
        "f1score": F1Score(task="binary").to(device=device),
    })
    checkpoint_path = rf".\run\{dataset_name}\{experiment_parameters.run_identifier}/"
    best_f1score_model_path = rf".\run\{dataset_name}\{experiment_parameters.run_identifier}/"
    best_loss_model_path = rf".\run\{dataset_name}\{experiment_parameters.run_identifier}/"
    non_improved_epoch = 0

    for epoch in range(experiment_parameters.epochs):
        epoch_index = epoch + experiment_parameters.load_epoch
        network, optimizer, gradient_scaler, total_step, learning_rate = execute_epoch(
            mode="train",
            dataloader=train_loader,
            device=device,
            log_path=log_path,
            network=network,
            optimizer=optimizer,
            total_step=total_step,
            learning_rate=learning_rate,
            criterion=calculate_multitask_change_detection_loss,
            metric_collection=metric_collection,
            epoch=epoch_index,
            warmup_learning_rates=warmup_learning_rates,
            gradient_scaler=gradient_scaler,
        )
        print("train success")

        if epoch_index + 1 >= experiment_parameters.evaluate_epoch:
            with torch.no_grad():
                network, optimizer, total_step, learning_rate, best_metrics, non_improved_epoch = execute_epoch(
                    mode="val",
                    dataloader=validation_loader,
                    device=device,
                    log_path=log_path,
                    network=network,
                    optimizer=optimizer,
                    total_step=total_step,
                    learning_rate=learning_rate,
                    criterion=calculate_multitask_change_detection_loss,
                    metric_collection=metric_collection,
                    epoch=epoch_index,
                    best_metrics=best_metrics,
                    checkpoint_path=checkpoint_path,
                    best_f1score_model_path=best_f1score_model_path,
                    best_loss_model_path=best_loss_model_path,
                    non_improved_epoch=non_improved_epoch,
                )
            print("val success")


if __name__ == "__main__":
    run_configured_experiment()
