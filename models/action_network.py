import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionNetwork(nn.Module):
    """
    Action network that outputs classification/action decisions.
    For classification: softmax over classes.
    For dynamic environments: softmax over actions.
    """

    def __init__(self, hidden_size, num_actions):
        """
        Initialize the action network.

        Args:
            hidden_size: Size of hidden state from core network
            num_actions: Number of possible actions/classes
        """
        super().__init__()
        self.num_actions = num_actions
        self.action_linear = nn.Linear(hidden_size, num_actions)

    def forward(self, hidden_state):
        """
        Forward pass through the action network.

        Args:
            hidden_state: Hidden state from core network

        Returns:
            Action logits
        """
        if len(hidden_state.shape) == 1:
            hidden_state = hidden_state.unsqueeze(0)

        logits = self.action_linear(hidden_state)
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def get_action(self, hidden_state):
        """
        Get action probabilities and sample an action.

        Args:
            hidden_state: Hidden state from core network

        Returns:
            Action probabilities, sampled action, log probability
        """
        logits = self.forward(hidden_state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return probs, action, log_prob

    def log_prob(self, action, hidden_state):
        """
        Compute log probability of an action given hidden state.

        Args:
            action: Action index
            hidden_state: Hidden state from core network

        Returns:
            Log probability
        """
        logits = self.forward(hidden_state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)

