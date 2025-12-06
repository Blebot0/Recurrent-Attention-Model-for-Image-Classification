import torch
import torch.nn as nn
import torch.nn.functional as F


class GlimpseNetwork(nn.Module):
    """
    Glimpse network that processes glimpse patches and location coordinates.
    Combines information from retina representation and location.
    """

    def __init__(self, glimpse_size, num_scales, hidden_size, output_size):
        """
        Initialize the glimpse network.

        Args:
            glimpse_size: Size of each glimpse patch
            num_scales: Number of resolution scales
            hidden_size: Hidden layer size for glimpse and location encoders
            output_size: Output size of the glimpse network
        """
        super().__init__()
        self.glimpse_size = glimpse_size
        self.num_scales = num_scales
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Input size: num_scales * glimpse_size * glimpse_size
        glimpse_input_size = num_scales * glimpse_size * glimpse_size
        location_input_size = 2  # (x, y) coordinates

        # Glimpse encoder: processes retina representation
        self.glimpse_encoder = nn.Sequential(
            nn.Linear(glimpse_input_size, hidden_size),
            nn.ReLU(),
        )

        # Location encoder: processes location coordinates
        self.location_encoder = nn.Sequential(
            nn.Linear(location_input_size, hidden_size),
            nn.ReLU(),
        )

        # Combine glimpse and location information
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.ReLU(),
        )

    def forward(self, glimpse_patch, location):
        """
        Forward pass through the glimpse network.

        Args:
            glimpse_patch: Flattened glimpse patches (shape: [batch, features] or [features])
            location: Location coordinates (x, y) (shape: [batch, 2] or [2])

        Returns:
            Glimpse feature vector
        """
        # Ensure inputs have batch dimension
        if len(glimpse_patch.shape) == 1:
            glimpse_patch = glimpse_patch.unsqueeze(0)
        if len(location.shape) == 1:
            location = location.unsqueeze(0)

        # Encode glimpse
        h_g = self.glimpse_encoder(glimpse_patch)

        # Encode location
        h_l = self.location_encoder(location)

        # Combine
        combined = torch.cat([h_g, h_l], dim=-1)
        output = self.combine(combined)

        # Remove batch dimension if input was single sample
        if output.shape[0] == 1:
            output = output.squeeze(0)

        return output

