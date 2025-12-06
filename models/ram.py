import torch
import torch.nn as nn
from models.glimpse_sensor import GlimpseSensor
from models.glimpse_network import GlimpseNetwork
from models.core_network import CoreNetwork
from models.location_network import LocationNetwork
from models.action_network import ActionNetwork
from models.value_network import ValueNetwork


class RecurrentAttentionModel(nn.Module):
    """
    Recurrent Attention Model (RAM) that processes images by
    adaptively selecting a sequence of regions to attend to.
    """

    def __init__(
        self,
        glimpse_size=8,
        num_scales=1,
        num_glimpses=6,
        hidden_size=256,
        num_actions=10,
        location_std=0.2,
        use_lstm=False,
    ):
        """
        Initialize the Recurrent Attention Model.

        Args:
            glimpse_size: Size of each glimpse patch (gw x gw)
            num_scales: Number of resolution scales in retina
            num_glimpses: Number of glimpses to take
            hidden_size: Size of hidden state
            num_actions: Number of possible actions/classes
            location_std: Standard deviation of location policy
            use_lstm: If True, use LSTM in core network; else use simple RNN
        """
        super().__init__()
        self.glimpse_size = glimpse_size
        self.num_scales = num_scales
        self.num_glimpses = num_glimpses
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.use_lstm = use_lstm

        # Glimpse sensor
        self.glimpse_sensor = GlimpseSensor(glimpse_size, num_scales)

        # Glimpse network
        glimpse_input_size = num_scales * glimpse_size * glimpse_size
        glimpse_hidden_size = 128
        self.glimpse_network = GlimpseNetwork(
            glimpse_size, num_scales, glimpse_hidden_size, hidden_size
        )

        # Core network
        self.core_network = CoreNetwork(hidden_size, hidden_size, use_lstm)

        # Location network
        self.location_network = LocationNetwork(hidden_size, location_std)

        # Action network
        self.action_network = ActionNetwork(hidden_size, num_actions)

        # Value network (for baseline)
        self.value_network = ValueNetwork(hidden_size)

    def forward(self, image, locations=None):
        """
        Forward pass through the model.

        Args:
            image: Input image tensor
            locations: Optional list of locations to use (for training)

        Returns:
            Dictionary containing actions, locations, log_probs, values
        """
        batch_size = 1
        if len(image.shape) > 2:
            batch_size = image.shape[0] if len(image.shape) == 3 else 1

        device = image.device
        
        # Handle single image vs batch
        if len(image.shape) == 4:
            # Batch: (B, C, H, W) or (B, 1, H, W)
            image = image[0]  # Process first image for now
        if len(image.shape) == 3 and image.shape[0] == 1:
            # (1, H, W) -> (H, W)
            image = image.squeeze(0)
        
        batch_size = 1
        hidden_state = None
        if self.use_lstm:
            hidden_state = (
                torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device),
            )

        actions = []
        sampled_locations = []
        location_log_probs = []
        action_log_probs = []
        values = []

        # First location is random
        if locations is None:
            location = torch.rand(2, device=device) * 2 - 1
        else:
            location = locations[0] if isinstance(locations[0], torch.Tensor) else torch.tensor(locations[0], device=device, dtype=torch.float32)

        for t in range(self.num_glimpses):
            # Extract glimpse
            glimpse_patch = self.glimpse_sensor.extract(image, location)
            if len(glimpse_patch.shape) == 1:
                glimpse_patch = glimpse_patch.unsqueeze(0)

            # Process glimpse
            glimpse_feature = self.glimpse_network(glimpse_patch, location)

            # Update hidden state
            if self.use_lstm:
                hidden_state = self.core_network(glimpse_feature, hidden_state)[1]
                current_hidden = hidden_state[0].squeeze(0)
            else:
                current_hidden = self.core_network(glimpse_feature, hidden_state)
                hidden_state = current_hidden

            # Get value estimate
            value = self.value_network(current_hidden)
            values.append(value)

            # Sample location for next step (except at last step)
            if t < self.num_glimpses - 1:
                if locations is None:
                    next_location, loc_log_prob = self.location_network.sample(
                        current_hidden
                    )
                else:
                    next_location = (
                        locations[t + 1]
                        if isinstance(locations[t + 1], torch.Tensor)
                        else torch.tensor(locations[t + 1], device=device, dtype=torch.float32)
                    )
                    loc_log_prob = self.location_network.log_prob(
                        next_location, current_hidden
                    )
                sampled_locations.append(next_location)
                location_log_probs.append(loc_log_prob)
                location = next_location

            # Get action (only at last step for classification)
            if t == self.num_glimpses - 1:
                _, action, action_log_prob = self.action_network.get_action(
                    current_hidden
                )
                actions.append(action)
                action_log_probs.append(action_log_prob)

        return {
            "actions": torch.stack(actions) if actions else None,
            "locations": torch.stack(sampled_locations) if sampled_locations else None,
            "location_log_probs": (
                torch.stack(location_log_probs) if location_log_probs else None
            ),
            "action_log_probs": (
                torch.stack(action_log_probs) if action_log_probs else None
            ),
            "values": torch.stack(values) if values else None,
            "hidden_state": current_hidden,
        }

    @torch.no_grad()
    def select_action(self, image):
        """
        Select action for a given image (inference mode).

        Args:
            image: Input image tensor

        Returns:
            Predicted action/class
        """
        output = self.forward(image)
        if output["actions"] is not None:
            return output["actions"].item()
        return None

    @torch.no_grad()
    def forward_with_glimpses(self, image, locations=None):
        """
        Forward pass that also returns glimpse patches for visualization.

        Args:
            image: Input image tensor
            locations: Optional list of locations to use (for training)

        Returns:
            Dictionary containing actions, locations, log_probs, values, and glimpse_patches
        """
        batch_size = 1
        if len(image.shape) > 2:
            batch_size = image.shape[0] if len(image.shape) == 3 else 1

        device = image.device
        
        # Handle single image vs batch
        if len(image.shape) == 4:
            image = image[0]
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        
        batch_size = 1
        hidden_state = None
        if self.use_lstm:
            hidden_state = (
                torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device),
            )

        actions = []
        sampled_locations = []
        location_log_probs = []
        action_log_probs = []
        values = []
        glimpse_patches_list = []  # Store patches for each glimpse
        all_locations = []  # Store all locations including first

        # First location is random
        if locations is None:
            location = torch.rand(2, device=device) * 2 - 1
        else:
            location = locations[0] if isinstance(locations[0], torch.Tensor) else torch.tensor(locations[0], device=device, dtype=torch.float32)
        
        all_locations.append(location)  # Store first location

        for t in range(self.num_glimpses):
            # Extract glimpse patches for visualization
            patches = self.glimpse_sensor.extract_patches(image, location)
            glimpse_patches_list.append(patches)
            
            # Extract glimpse (flattened for processing)
            glimpse_patch = self.glimpse_sensor.extract(image, location)
            if len(glimpse_patch.shape) == 1:
                glimpse_patch = glimpse_patch.unsqueeze(0)

            # Process glimpse
            glimpse_feature = self.glimpse_network(glimpse_patch, location)

            # Update hidden state
            if self.use_lstm:
                hidden_state = self.core_network(glimpse_feature, hidden_state)[1]
                current_hidden = hidden_state[0].squeeze(0)
            else:
                current_hidden = self.core_network(glimpse_feature, hidden_state)
                hidden_state = current_hidden

            # Get value estimate
            value = self.value_network(current_hidden)
            values.append(value)

            # Sample location for next step (except at last step)
            if t < self.num_glimpses - 1:
                if locations is None:
                    next_location, loc_log_prob = self.location_network.sample(
                        current_hidden
                    )
                else:
                    next_location = (
                        locations[t + 1]
                        if isinstance(locations[t + 1], torch.Tensor)
                        else torch.tensor(locations[t + 1], device=device, dtype=torch.float32)
                    )
                    loc_log_prob = self.location_network.log_prob(
                        next_location, current_hidden
                    )
                sampled_locations.append(next_location)
                all_locations.append(next_location)  # Store for visualization
                location_log_probs.append(loc_log_prob)
                location = next_location

            # Get action (only at last step for classification)
            if t == self.num_glimpses - 1:
                _, action, action_log_prob = self.action_network.get_action(
                    current_hidden
                )
                actions.append(action)
                action_log_probs.append(action_log_prob)

        return {
            "actions": torch.stack(actions) if actions else None,
            "locations": torch.stack(sampled_locations) if sampled_locations else None,
            "all_locations": torch.stack(all_locations) if all_locations else None,  # All locations including first
            "location_log_probs": (
                torch.stack(location_log_probs) if location_log_probs else None
            ),
            "action_log_probs": (
                torch.stack(action_log_probs) if action_log_probs else None
            ),
            "values": torch.stack(values) if values else None,
            "hidden_state": current_hidden,
            "glimpse_patches": glimpse_patches_list,  # List of lists of patches
        }

