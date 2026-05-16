import torch
import torch.nn as nn

from config.hyperparameters import experiment_parameters


class SoftDiceLoss(nn.Module):
    def __init__(self, reduce_over_batch=True):
        super().__init__()
        self.reduce_over_batch = reduce_over_batch

    def soft_dice_coefficient(self, prediction, target):
        smooth = 1e-5
        prediction = prediction.float()
        target = target.float()
        if self.reduce_over_batch:
            prediction_area = torch.sum(prediction)
            target_area = torch.sum(target)
            intersection = torch.sum(target * prediction)
        else:
            prediction_area = prediction.sum(1).sum(1).sum(1)
            target_area = target.sum(1).sum(1).sum(1)
            intersection = (target * prediction).sum(1).sum(1).sum(1)
        return ((2.0 * intersection + smooth) / (target_area + prediction_area + smooth)).mean()

    def forward(self, prediction, target):
        return 1 - self.soft_dice_coefficient(prediction, target)


class ChangeBranchCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss()

    def forward(self, logits, labels):
        labels = labels.float()
        probability = torch.sigmoid(logits)
        dice_term = self.dice_loss(probability, labels)
        bce_term = self.binary_cross_entropy(logits, labels)

        if not torch.isfinite(dice_term):
            raise ValueError(f"Dice loss is not finite: {dice_term}")
        if not torch.isfinite(bce_term):
            raise ValueError(f"BCEWithLogits loss is not finite: {bce_term}")

        return dice_term + bce_term


def combine_objectives_with_dynamic_weights(losses, initial_weights=None):
    if initial_weights is None:
        initial_weights = [1, experiment_parameters.beta, experiment_parameters.beta, experiment_parameters.beta]

    assert all(isinstance(loss, torch.Tensor) for loss in losses), "All losses must be tensors."

    scalar_losses = [torch.mean(loss) for loss in losses]
    for loss_index, scalar_loss in enumerate(scalar_losses):
        if not torch.isfinite(scalar_loss):
            raise ValueError(f"Loss {loss_index} is not finite: {scalar_loss}")

    reciprocal_losses = [1.0 / (scalar_loss.detach() + 1.0) for scalar_loss in scalar_losses]
    weighted_terms = [
        initial_weight * reciprocal_loss
        for initial_weight, reciprocal_loss in zip(initial_weights, reciprocal_losses)
    ]
    weight_sum = sum(weighted_terms)

    if not torch.isfinite(weight_sum) or float(weight_sum) == 0.0:
        normalizer = float(sum(initial_weights))
        normalized_weights = [float(weight) / normalizer for weight in initial_weights]
    else:
        normalized_weights = [weight / weight_sum for weight in weighted_terms]

    combined_losses = [weight * loss for weight, loss in zip(normalized_weights, scalar_losses)]

    for loss_index, combined_loss in enumerate(combined_losses):
        if not torch.isfinite(combined_loss):
            raise ValueError(f"Combined loss {loss_index} is not finite: {combined_loss}; raw losses={scalar_losses}")

    return combined_losses


def calculate_direction_field_loss(predicted_flux, target_flux, weight_matrix):
    device = predicted_flux.device
    weight_matrix = weight_matrix.cuda(device)
    target_flux = target_flux.cuda(device)

    normalized_target_flux = 0.999999 * target_flux / (target_flux.norm(p=2, dim=1, keepdim=True) + 1e-9)
    weighted_residual = weight_matrix.unsqueeze(1) * (predicted_flux - normalized_target_flux) ** 2
    magnitude_loss = weighted_residual.sum()

    normalized_prediction_flux = 0.999999 * predicted_flux / (predicted_flux.norm(p=2, dim=1, keepdim=True) + 1e-9)
    cosine_similarity = torch.sum(normalized_prediction_flux * normalized_target_flux, dim=1).clamp(-0.999999, 0.999999)
    angular_loss = (torch.acos(cosine_similarity)) ** 2
    direction_loss = angular_loss.sum() + magnitude_loss

    if not torch.isfinite(direction_loss):
        raise ValueError(f"Segmentation direction loss is not finite: {direction_loss}")
    return direction_loss


def calculate_multitask_change_detection_loss(
        change_logits,
        t1_direction_logits,
        t2_direction_logits,
        all_direction_logits,
        change_label,
        t1_direction_target,
        t2_direction_target,
        all_direction_target,
        t1_weight_matrix,
        t2_weight_matrix,
        all_weight_matrix):
    change_logits = change_logits.squeeze(1) if len(change_logits.shape) > 3 else change_logits
    change_label = change_label.squeeze(1) if len(change_label.shape) > 3 else change_label
    change_label = change_label.float()

    change_loss = ChangeBranchCriterion()(change_logits, change_label)
    t1_direction_loss = calculate_direction_field_loss(t1_direction_logits, t1_direction_target, t1_weight_matrix)
    t2_direction_loss = calculate_direction_field_loss(t2_direction_logits, t2_direction_target, t2_weight_matrix)
    all_direction_loss = calculate_direction_field_loss(all_direction_logits, all_direction_target, all_weight_matrix)

    return combine_objectives_with_dynamic_weights(
        [change_loss, t1_direction_loss, t2_direction_loss, all_direction_loss]
    )
