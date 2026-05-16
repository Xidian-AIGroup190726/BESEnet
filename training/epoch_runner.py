import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config.hyperparameters import experiment_parameters

os.environ["ALBUMENTATIONS_SKIP_VERSION_CHECK"] = "1"


def build_boundary_target(mask):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    eroded_mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return ((mask - eroded_mask) > 0).float()


def persist_model_state(model, path, epoch, mode, optimizer=None):
    assert mode in ["checkpoint", "loss", "f1score"], "mode should be 'checkpoint', 'loss', or 'f1score'"
    Path(path).mkdir(parents=True, exist_ok=True)
    saved_at = time.asctime(time.localtime(time.time()))

    if mode == "checkpoint":
        file_path = os.path.join(path, rf"checkpoint_epoch{epoch}.pth")
        state_dict = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
        }
    else:
        file_path = os.path.join(path, rf"best_{mode}_epoch{epoch}.pth")
        state_dict = model.state_dict()

    torch.save(state_dict, file_path)
    print(f"best {mode} model {epoch} saved at {saved_at}!")


def move_batch_to_device(batch, device):
    (
        t1_image,
        t2_image,
        change_label,
        t1_direction_label,
        t2_direction_label,
        all_direction_label,
        t1_weight,
        t2_weight,
        all_weight,
        sample_name,
    ) = batch
    return (
        t1_image.float().to(device),
        t2_image.float().to(device),
        change_label.float().to(device),
        t1_direction_label.float().to(device),
        t2_direction_label.float().to(device),
        all_direction_label.float().to(device),
        t1_weight.float().to(device),
        t2_weight.float().to(device),
        all_weight.float().to(device),
        sample_name,
    )


def evaluate_output_integrity(epoch, t1_image, t2_image, predictions):
    tensor_names = ["change_logits", "t1_direction_logits", "t2_direction_logits", "all_direction_logits"]
    for tensor_name, tensor in zip(tensor_names, predictions[:4]):
        if not torch.isfinite(tensor).all():
            print(f"\n{tensor_name} has NaN/Inf")
            print("epoch:", epoch)
            print("img1 min/max:", t1_image.min().item(), t1_image.max().item())
            print("img2 min/max:", t2_image.min().item(), t2_image.max().item())
            raise ValueError(f"{tensor_name} has NaN/Inf before loss")


def calculate_epoch_loss(mode, epoch, network, batch_tensors, criterion):
    (
        t1_image,
        t2_image,
        change_label,
        t1_direction_label,
        t2_direction_label,
        all_direction_label,
        t1_weight,
        t2_weight,
        all_weight,
        _,
    ) = batch_tensors
    boundary_target = build_boundary_target(change_label).to(t1_image.device)
    predictions = network(t1_image, t2_image)

    if mode == "val":
        evaluate_output_integrity(epoch, t1_image, t2_image, predictions)

    change_logits, t1_direction_logits, t2_direction_logits, all_direction_logits, boundary_logits = predictions
    objective_terms = criterion(
        change_logits,
        t1_direction_logits,
        t2_direction_logits,
        all_direction_logits,
        change_label,
        t1_direction_label,
        t2_direction_label,
        all_direction_label,
        t1_weight,
        t2_weight,
        all_weight,
    )

    if boundary_logits.shape[-2:] != boundary_target.shape[-2:]:
        boundary_logits = F.interpolate(
            boundary_logits,
            size=boundary_target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    boundary_loss = nn.BCEWithLogitsLoss()(boundary_logits, boundary_target.float())
    total_loss = sum(objective_terms)
    if mode == "train" and epoch >= 75:
        total_loss = total_loss + 0.1 * boundary_loss
    if mode == "val" and epoch >= 100:
        total_loss = total_loss + 0.001 * boundary_loss

    return predictions, objective_terms, total_loss


def write_batch_log(log_path, mode, loss, objective_terms, metrics, learning_rate, total_step, epoch):
    with open(log_path, "a", encoding="utf-8") as logger:
        logger.write(f"""
            {mode} loss: {loss.item()}
            {mode} iou: {metrics["iou"]}
            {mode} precision: {metrics["precision"]}
            {mode} recall: {metrics["recall"]}
            {mode} f1score: {metrics["f1score"]}
            learning rate: {learning_rate}
            {mode} loss_change: {objective_terms[0].item()}
            {mode} loss_seg1: {objective_terms[1].item()}
            {mode} loss_seg2: {objective_terms[2].item()}
            {mode} loss_seg_all: {objective_terms[3].item()}
            step: {total_step}
            epoch: {epoch}
            """)


def write_epoch_log(log_path, mode, epoch, epoch_metrics, epoch_loss):
    with open(log_path, "a", encoding="utf-8") as logger:
        for metric_name, metric_value in epoch_metrics.items():
            logger.write(f"epoch_{mode}_{metric_name}: {metric_value}\nepoch: {epoch}\n")
        logger.write(f"epoch_{mode}_loss: {epoch_loss}\nepoch: {epoch}\n")


def update_validation_state(
        epoch,
        epoch_metrics,
        epoch_loss,
        best_metrics,
        network,
        optimizer,
        learning_rate,
        checkpoint_path,
        best_f1score_model_path,
        best_loss_model_path,
        non_improved_epoch):
    if epoch_metrics["f1score"] > best_metrics["best_f1score"]:
        non_improved_epoch = 0
        best_metrics["best_f1score"] = epoch_metrics["f1score"]
        if experiment_parameters.save_best_model:
            persist_model_state(network, best_f1score_model_path, epoch, "f1score")
    elif epoch_loss < best_metrics["lowest loss"]:
        best_metrics["lowest loss"] = epoch_loss
        if experiment_parameters.save_best_model:
            persist_model_state(network, best_loss_model_path, epoch, "loss")
    else:
        non_improved_epoch += 1
        if non_improved_epoch == experiment_parameters.patience:
            learning_rate *= experiment_parameters.factor
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            non_improved_epoch = 0

    if (epoch + 1) % experiment_parameters.save_interval == 0 and experiment_parameters.save_checkpoint:
        persist_model_state(network, checkpoint_path, epoch, "checkpoint", optimizer=optimizer)

    return learning_rate, best_metrics, non_improved_epoch


def execute_epoch(
        mode,
        dataloader,
        device,
        log_path,
        network,
        optimizer,
        total_step,
        learning_rate,
        criterion,
        metric_collection,
        epoch,
        warmup_learning_rates=None,
        gradient_scaler=None,
        best_metrics=None,
        checkpoint_path=None,
        best_f1score_model_path=None,
        best_loss_model_path=None,
        non_improved_epoch=None):
    assert mode in ["train", "val"], "mode should be train, val"

    network.train() if mode == "train" else network.eval()
    epoch_loss = 0.0
    batch_progress = 0
    progress_bar = tqdm(dataloader)
    iteration_count = len(dataloader)
    sampled_batch_index = np.random.randint(low=0, high=iteration_count)

    if epoch == 0:
        trainable_parameters = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
        print(f"trainable parameters: {trainable_parameters}")

    for batch_index, batch in enumerate(progress_bar):
        progress_bar.set_description(
            f"epoch {epoch} info {batch_progress} - {batch_progress + experiment_parameters.batch_size}"
        )
        batch_progress += experiment_parameters.batch_size
        total_step += 1

        if mode == "train":
            optimizer.zero_grad()
            if total_step < experiment_parameters.warm_up_step and warmup_learning_rates is not None:
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = warmup_learning_rates[total_step]

        batch_tensors = move_batch_to_device(batch, device)

        if mode == "train":
            with torch.cuda.amp.autocast():
                predictions, objective_terms, loss = calculate_epoch_loss(mode, epoch, network, batch_tensors, criterion)
            if gradient_scaler is not None:
                gradient_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), experiment_parameters.max_norm, norm_type=2)
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), experiment_parameters.max_norm, norm_type=2)
                optimizer.step()
        else:
            predictions, objective_terms, loss = calculate_epoch_loss(mode, epoch, network, batch_tensors, criterion)

        epoch_loss += loss.item()
        change_probability = torch.sigmoid(predictions[0]).float()
        change_label = batch_tensors[2]
        batch_metrics = metric_collection.forward(change_probability, change_label.int().unsqueeze(1))

        if batch_index == sampled_batch_index:
            current_learning_rate = optimizer.param_groups[0]["lr"] if optimizer is not None else learning_rate
            write_batch_log(log_path, mode, loss, objective_terms, batch_metrics, current_learning_rate, total_step, epoch)

        del batch_tensors

    epoch_metrics = metric_collection.compute()
    epoch_loss /= iteration_count
    write_epoch_log(log_path, mode, epoch, epoch_metrics, epoch_loss)
    metric_collection.reset()

    if mode == "val":
        learning_rate, best_metrics, non_improved_epoch = update_validation_state(
            epoch,
            epoch_metrics,
            epoch_loss,
            best_metrics,
            network,
            optimizer,
            learning_rate,
            checkpoint_path,
            best_f1score_model_path,
            best_loss_model_path,
            non_improved_epoch,
        )
        return network, optimizer, total_step, learning_rate, best_metrics, non_improved_epoch

    return network, optimizer, gradient_scaler, total_step, learning_rate
