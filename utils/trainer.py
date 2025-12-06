import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class RAMTrainer:
    """
    Trainer for Recurrent Attention Model using REINFORCE for location
    network and supervised learning for action network.
    """

    def __init__(
        self,
        model,
        lr=1e-3,
        momentum=0.9,
        location_weight=1.0,
        action_weight=1.0,
        value_weight=0.5,
        device="cpu",
    ):
        """
        Initialize the trainer.

        Args:
            model: RecurrentAttentionModel instance
            lr: Learning rate
            momentum: Momentum for SGD
            location_weight: Weight for location policy loss
            action_weight: Weight for action loss
            value_weight: Weight for value loss
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.location_weight = location_weight
        self.action_weight = action_weight
        self.value_weight = value_weight

        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum
        )

        # Loss functions
        self.action_criterion = nn.CrossEntropyLoss()

    def train_step_classification(
        self, images, labels, num_glimpses=None
    ):
        """
        Train one step on classification task.

        Args:
            images: Batch of images
            labels: Ground truth labels
            num_glimpses: Number of glimpses (overrides model default)

        Returns:
            Dictionary with losses and metrics
        """
        self.model.train()

        if num_glimpses is None:
            num_glimpses = self.model.num_glimpses

        batch_size = images.shape[0] if len(images.shape) > 2 else 1
        if len(images.shape) == 2:
            images = images.unsqueeze(0)

        images = images.to(self.device)
        labels = labels.to(self.device)

        # Accumulate losses and metrics over batch
        batch_total_loss = 0.0
        batch_action_loss = 0.0
        batch_location_loss = 0.0
        batch_value_loss = 0.0
        batch_correct = 0

        # Process each image individually
        for i in range(batch_size):
            self.optimizer.zero_grad()
            
            image = images[i]
            if len(image.shape) == 3 and image.shape[0] == 1:
                image = image.squeeze(0)  # Remove channel dimension if single channel
            label = labels[i]  # Get scalar label for CrossEntropyLoss
            if len(label.shape) > 0 and label.shape[0] == 1:
                label = label.squeeze()

            # Forward pass
            outputs = self.model(image)

            # Compute losses
            total_loss = 0.0

            # Action loss (supervised learning)
            # Get action logits from last hidden state
            hidden_state = outputs["hidden_state"]
            if len(hidden_state.shape) == 1:
                hidden_state = hidden_state.unsqueeze(0)

            action_logits = self.model.action_network(hidden_state)
            action_loss = self.action_criterion(action_logits, label)
            batch_action_loss += action_loss.item()
            total_loss += self.action_weight * action_loss

            # Location loss (REINFORCE)
            if outputs["location_log_probs"] is not None:
                # Compute rewards: 1 if correct, 0 otherwise
                predicted = torch.argmax(action_logits, dim=-1)
                rewards = (predicted == label).float()

                # We have num_glimpses - 1 location decisions
                num_locations = outputs["location_log_probs"].shape[0]

                # Compute cumulative rewards for each location decision
                # Rewards is scalar, expand to [num_locations]
                if rewards.dim() == 0:
                    cumulative_rewards = rewards.unsqueeze(0).expand(num_locations)
                else:
                    cumulative_rewards = rewards.unsqueeze(-1).expand(-1, num_locations)

                # Value baseline: use values for first num_locations timesteps
                if outputs["values"] is not None:
                    # Stack values and take first num_locations
                    all_values = torch.stack([outputs["values"][i] for i in range(num_locations)])
                    baselines = all_values.squeeze()
                    if len(baselines.shape) == 0:
                        baselines = baselines.unsqueeze(0)
                    # Ensure baselines has shape [num_locations]
                    advantages = cumulative_rewards - baselines.detach()
                else:
                    advantages = cumulative_rewards

                # REINFORCE loss
                location_loss = -torch.mean(
                    outputs["location_log_probs"] * advantages
                )
                batch_location_loss += location_loss.item()
                total_loss += self.location_weight * location_loss

                # Value loss: only for location timesteps
                if outputs["values"] is not None:
                    # Ensure shapes match for MSE loss
                    value_loss = nn.MSELoss()(baselines, cumulative_rewards)
                    batch_value_loss += value_loss.item()
                    total_loss += self.value_weight * value_loss

            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            batch_total_loss += total_loss.item()

            # Compute accuracy
            with torch.no_grad():
                predicted = torch.argmax(action_logits, dim=-1)
                batch_correct += (predicted == label).float().item()

        # Return average losses over batch
        losses = {
            "total_loss": batch_total_loss / batch_size,
            "action_loss": batch_action_loss / batch_size,
            "accuracy": batch_correct / batch_size,
        }
        
        if batch_location_loss > 0:
            losses["location_loss"] = batch_location_loss / batch_size
        if batch_value_loss > 0:
            losses["value_loss"] = batch_value_loss / batch_size

        return losses

    def train_step_reinforce(
        self, images, rewards, num_glimpses=None
    ):
        """
        Train one step using pure REINFORCE (for dynamic environments).

        Args:
            images: Batch of images
            rewards: Rewards received
            num_glimpses: Number of glimpses

        Returns:
            Dictionary with losses
        """
        self.model.train()
        self.optimizer.zero_grad()

        if num_glimpses is None:
            num_glimpses = self.model.num_glimpses

        batch_size = images.shape[0] if len(images.shape) > 2 else 1
        if len(images.shape) == 2:
            images = images.unsqueeze(0)

        images = images.to(self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Forward pass
        outputs = self.model(images[0] if batch_size == 1 else images)

        total_loss = 0.0
        losses = {}

        # Location loss (REINFORCE)
        if outputs["location_log_probs"] is not None:
            cumulative_rewards = rewards.unsqueeze(-1).expand(
                -1, outputs["location_log_probs"].shape[0]
            )

            # Value baseline
            if outputs["values"] is not None:
                baselines = outputs["values"].squeeze()
                if len(baselines.shape) == 1:
                    baselines = baselines.unsqueeze(0)
                advantages = cumulative_rewards - baselines.detach()
            else:
                advantages = cumulative_rewards

            location_loss = -torch.mean(
                outputs["location_log_probs"] * advantages
            )
            losses["location_loss"] = location_loss.item()
            total_loss += self.location_weight * location_loss

            # Value loss
            if outputs["values"] is not None:
                value_loss = nn.MSELoss()(baselines, cumulative_rewards)
                losses["value_loss"] = value_loss.item()
                total_loss += self.value_weight * value_loss

        # Action loss (REINFORCE)
        if outputs["action_log_probs"] is not None:
            advantages = rewards - (
                outputs["values"][-1].squeeze() if outputs["values"] is not None else 0
            )
            action_loss = -torch.mean(outputs["action_log_probs"] * advantages)
            losses["action_loss"] = action_loss.item()
            total_loss += self.action_weight * action_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        losses["total_loss"] = total_loss.item()

        return losses

    @torch.no_grad()
    def evaluate(self, dataloader, num_glimpses=None):
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            num_glimpses: Number of glimpses

        Returns:
            Dictionary with accuracy and other metrics
        """
        self.model.eval()

        if num_glimpses is None:
            num_glimpses = self.model.num_glimpses

        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if len(images.shape) == 2:
                images = images.unsqueeze(0)

            # Get predictions
            predictions = []
            for i in range(images.shape[0]):
                pred = self.model.select_action(images[i])
                predictions.append(pred)

            predictions = torch.tensor(predictions, device=self.device, dtype=torch.long)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": total}

