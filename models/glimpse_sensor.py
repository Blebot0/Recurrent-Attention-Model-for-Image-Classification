import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


class GlimpseSensor:
    """
    Extracts retina-like representation from images around a given location.
    Creates multiple resolution patches centered at the location.
    """

    def __init__(self, glimpse_size, num_scales):
        """
        Initialize the glimpse sensor.

        Args:
            glimpse_size: Size of each glimpse patch (gw x gw)
            num_scales: Number of different resolution scales
        """
        self.glimpse_size = glimpse_size
        self.num_scales = num_scales

    def extract(self, image, location):
        """
        Extract retina-like representation around location.

        Args:
            image: Input image tensor of shape (H, W), (C, H, W), or (1, H, W)
            location: Location coordinates (x, y) in range [-1, 1]

        Returns:
            Concatenated glimpse patches as a flattened vector
        """
        # Handle different input shapes
        if len(image.shape) == 4:
            # Batch dimension, take first
            image = image[0]
        if len(image.shape) == 3:
            # (C, H, W) or (1, H, W)
            if image.shape[0] == 1:
                image = image.squeeze(0)
            else:
                # Convert to grayscale if RGB
                image = image.mean(dim=0)
        # Now image should be (H, W)

        h, w = image.shape[-2:]
        patches = []

        # Convert location from [-1, 1] to pixel coordinates
        # (0, 0) is center, (-1, -1) is top-left
        # Handle CUDA tensors by extracting values properly
        loc_x = location[0].item() if isinstance(location[0], torch.Tensor) else location[0]
        loc_y = location[1].item() if isinstance(location[1], torch.Tensor) else location[1]
        x_pixel = ((loc_x + 1) / 2.0) * w
        y_pixel = ((loc_y + 1) / 2.0) * h

        for scale in range(self.num_scales):
            patch_size = self.glimpse_size * (2 ** scale)
            patch = self._extract_patch(image, x_pixel, y_pixel, patch_size)
            # Resize to glimpse_size x glimpse_size
            # Keep the patch on the same device as the input image
            patch = F.interpolate(
                patch.unsqueeze(0).unsqueeze(0),
                size=(self.glimpse_size, self.glimpse_size),
                mode="bilinear",
                align_corners=False,
            )
            patches.append(patch.squeeze())

        # Concatenate all patches
        glimpse = torch.cat(patches, dim=0)
        return glimpse.flatten()

    def _extract_patch(self, image, x, y, patch_size):
        """
        Extract a patch of given size centered at (x, y).

        Args:
            image: Input image tensor
            x, y: Center coordinates in pixels
            patch_size: Size of the patch to extract

        Returns:
            Extracted patch
        """
        h, w = image.shape[-2:]
        half_size = patch_size // 2

        # Calculate patch boundaries
        x_min = max(0, int(x - half_size))
        x_max = min(w, int(x + half_size))
        y_min = max(0, int(y - half_size))
        y_max = min(h, int(y + half_size))

        # Extract patch
        patch = image[y_min:y_max, x_min:x_max]

        # Pad if necessary
        pad_h = patch_size - (y_max - y_min)
        pad_w = patch_size - (x_max - x_min)
        pad_top = max(0, int(y - half_size) - y_min)
        pad_bottom = pad_h - pad_top
        pad_left = max(0, int(x - half_size) - x_min)
        pad_right = pad_w - pad_left

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            patch = F.pad(
                patch.unsqueeze(0),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            patch = patch.squeeze(0)

        return patch

    def extract_patches(self, image, location):
        """
        Extract retina-like representation around location, returning patches
        in a visualizable format (not flattened).

        Args:
            image: Input image tensor of shape (H, W), (C, H, W), or (1, H, W)
            location: Location coordinates (x, y) in range [-1, 1]

        Returns:
            List of glimpse patches, each of shape (glimpse_size, glimpse_size)
        """
        # Handle different input shapes
        if len(image.shape) == 4:
            image = image[0]
        if len(image.shape) == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            else:
                image = image.mean(dim=0)
        # Now image should be (H, W)

        h, w = image.shape[-2:]
        patches = []

        # Convert location from [-1, 1] to pixel coordinates
        loc_x = location[0].item() if isinstance(location[0], torch.Tensor) else location[0]
        loc_y = location[1].item() if isinstance(location[1], torch.Tensor) else location[1]
        x_pixel = ((loc_x + 1) / 2.0) * w
        y_pixel = ((loc_y + 1) / 2.0) * h

        for scale in range(self.num_scales):
            patch_size = self.glimpse_size * (2 ** scale)
            patch = self._extract_patch(image, x_pixel, y_pixel, patch_size)
            # Resize to glimpse_size x glimpse_size
            patch = F.interpolate(
                patch.unsqueeze(0).unsqueeze(0),
                size=(self.glimpse_size, self.glimpse_size),
                mode="bilinear",
                align_corners=False,
            )
            patches.append(patch.squeeze())

        return patches

