import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import RecurrentAttentionModel
from utils.datasets import (
    get_mnist_datasets,
)
from utils.trainer import RAMTrainer


def visualize_glimpses(image, output, save_path=None):
    """
    Visualize glimpse patches and attention locations for a single image.
    
    Args:
        image: Input image tensor (H, W) or (1, H, W)
        output: Output dictionary from forward_with_glimpses
        save_path: Path to save the visualization (optional)
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        image_np = image.cpu().numpy()
    else:
        image_np = image
    
    # Use all_locations if available (includes first location), otherwise fall back to locations
    all_locations_tensor = output.get("all_locations", output["locations"])
    locations = output["locations"]
    glimpse_patches = output["glimpse_patches"]
    predicted_class = output["actions"].item() if output["actions"] is not None else None
    
    h, w = image_np.shape
    
    # Create figure with subplots
    num_glimpses = len(glimpse_patches)
    fig = plt.figure(figsize=(15, 4 + num_glimpses * 2))
    
    # Plot 1: Original image with location markers
    ax1 = plt.subplot(2, num_glimpses + 1, 1)
    ax1.imshow(image_np, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Original Image\nPredicted: {predicted_class}', fontsize=10)
    ax1.axis('off')
    
    # Plot locations on the image
    all_locations = []
    if all_locations_tensor is not None and len(all_locations_tensor) > 0:
        # Convert locations from [-1, 1] to pixel coordinates
        for loc in all_locations_tensor:
            loc_x = loc[0].item() if isinstance(loc[0], torch.Tensor) else loc[0]
            loc_y = loc[1].item() if isinstance(loc[1], torch.Tensor) else loc[1]
            x_pixel = ((loc_x + 1) / 2.0) * w
            y_pixel = ((loc_y + 1) / 2.0) * h
            all_locations.append((x_pixel, y_pixel))
    
    # Draw location markers
    colors = plt.cm.tab10(np.linspace(0, 1, num_glimpses))
    for i, (x, y) in enumerate(all_locations):
        circle = plt.Circle((x, y), 3, color=colors[i], fill=True, alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(x + 5, y - 5, f'G{i+1}', color=colors[i], fontsize=8, weight='bold')
    
    # Draw connection lines between glimpses
    if len(all_locations) > 1:
        for i in range(len(all_locations) - 1):
            x1, y1 = all_locations[i]
            x2, y2 = all_locations[i + 1]
            ax1.plot([x1, x2], [y1, y2], color=colors[i], alpha=0.3, linewidth=1, linestyle='--')
    
    # Plot 2: Glimpse patches for each time step
    for t in range(num_glimpses):
        ax = plt.subplot(2, num_glimpses + 1, t + 2)
        patches_t = glimpse_patches[t]
        
        # If multiple scales, show them side by side
        if len(patches_t) > 1:
            # Concatenate patches horizontally
            combined = torch.cat(patches_t, dim=1).cpu().numpy()
            ax.imshow(combined, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Glimpse {t+1}\n({len(patches_t)} scales)', fontsize=9)
        else:
            patch_np = patches_t[0].cpu().numpy() if isinstance(patches_t[0], torch.Tensor) else patches_t[0]
            ax.imshow(patch_np, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Glimpse {t+1}', fontsize=9)
        ax.axis('off')
    
    # Plot 3: Location sequence visualization
    ax2 = plt.subplot(2, num_glimpses + 1, num_glimpses + 2)
    if all_locations_tensor is not None and len(all_locations_tensor) > 0:
        locs_np = []
        for loc in all_locations_tensor:
            loc_x = loc[0].item() if isinstance(loc[0], torch.Tensor) else loc[0]
            loc_y = loc[1].item() if isinstance(loc[1], torch.Tensor) else loc[1]
            locs_np.append([loc_x, loc_y])
        locs_np = np.array(locs_np)
        
        ax2.plot(locs_np[:, 0], locs_np[:, 1], 'o-', markersize=8, linewidth=2, alpha=0.7)
        ax2.scatter(locs_np[0, 0], locs_np[0, 1], s=100, c='green', marker='s', 
                   label='Start', zorder=5, edgecolors='black', linewidths=1)
        ax2.scatter(locs_np[-1, 0], locs_np[-1, 1], s=100, c='red', marker='s', 
                   label='End', zorder=5, edgecolors='black', linewidths=1)
        for i, (x, y) in enumerate(locs_np):
            ax2.text(x + 0.02, y + 0.02, f'{i+1}', fontsize=8)
        ax2.set_xlabel('X location (normalized)')
        ax2.set_ylabel('Y location (normalized)')
        ax2.set_title('Attention Trajectory')
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(
    model_path,
    dataset_type="mnist",
    image_size=28,
    num_glimpses=6,
    glimpse_size=8,
    num_scales=1,
    hidden_size=256,
    device="cpu",
    data_dir="./data",
    visualize=False,
    num_visualize=5,
    save_dir="./plots",
):
    """
    Evaluate a trained RAM model.

    Args:
        model_path: Path to saved model
        dataset_type: Type of dataset
        image_size: Size of images
        num_glimpses: Number of glimpses
        glimpse_size: Size of glimpse patches
        num_scales: Number of resolution scales
        hidden_size: Hidden state size
        device: Device to use
        data_dir: Data directory
    """
    # Load dataset
    if dataset_type == "mnist":
        _, test_dataset = get_mnist_datasets(
            data_dir=data_dir, image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create model
    model = RecurrentAttentionModel(
        glimpse_size=glimpse_size,
        num_scales=num_scales,
        num_glimpses=num_glimpses,
        hidden_size=hidden_size,
        num_actions=10,
        location_std=0.2,
        use_lstm=False,
    )

    # Load model weights
    # Note: weights_only=False is needed to load checkpoints with metadata (config, history, etc.)
    # This is safe for checkpoints created by your own training runs
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both old format (state_dict only) and new format (full checkpoint)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'test_accuracy' in checkpoint:
            print(f"Checkpoint test accuracy: {checkpoint['test_accuracy']:.4f}")
        if 'config' in checkpoint:
            print(f"Model config: {checkpoint['config']}")
    else:
        # Old format: just state_dict
        model.load_state_dict(checkpoint)
        print("Loaded model weights (old format)")
    
    model.to(device)

    # Create trainer for evaluation
    trainer = RAMTrainer(model, device=device)

    # Evaluate
    print(f"\nEvaluating on {dataset_type} test set...")
    metrics = trainer.evaluate(test_loader, num_glimpses=num_glimpses)

    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
    
    # Visualize glimpses if requested
    if visualize:
        print(f"\nVisualizing glimpses for {num_visualize} sample images...")
        model.eval()
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        visualize_count = 0
        for batch_idx, (images, labels) in enumerate(test_loader):
            if visualize_count >= num_visualize:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Process each image in the batch
            for img_idx in range(images.shape[0]):
                if visualize_count >= num_visualize:
                    break
                    
                image = images[img_idx:img_idx+1]
                label = labels[img_idx].item()
                
                # Get output with glimpses
                output = model.forward_with_glimpses(image)
                predicted = output["actions"].item() if output["actions"] is not None else None
                
                # Prepare image for visualization
                if len(image.shape) == 4:
                    vis_image = image[0]
                else:
                    vis_image = image
                if len(vis_image.shape) == 3 and vis_image.shape[0] == 1:
                    vis_image = vis_image.squeeze(0)
                
                # Save visualization
                viz_filename = save_path / f"glimpse_vis_sample_{visualize_count}_true_{label}_pred_{predicted}.png"
                visualize_glimpses(vis_image, output, save_path=viz_filename)
                
                print(f"  Sample {visualize_count + 1}: True={label}, Predicted={predicted}")
                visualize_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAM model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist"],
        help="Dataset type",
    )
    parser.add_argument("--image_size", type=int, default=28, help="Image size")
    parser.add_argument(
        "--num_glimpses", type=int, default=6, help="Number of glimpses"
    )
    parser.add_argument(
        "--glimpse_size", type=int, default=8, help="Glimpse patch size"
    )
    parser.add_argument(
        "--num_scales", type=int, default=1, help="Number of resolution scales"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden state size"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize glimpse patches for sample images",
    )
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=5,
        help="Number of sample images to visualize",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./plots",
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    evaluate_model(
        model_path=args.model_path,
        dataset_type=args.dataset,
        image_size=args.image_size,
        num_glimpses=args.num_glimpses,
        glimpse_size=args.glimpse_size,
        num_scales=args.num_scales,
        hidden_size=args.hidden_size,
        device=device,
        data_dir=args.data_dir,
        visualize=args.visualize,
        num_visualize=args.num_visualize,
        save_dir=args.save_dir,
    )

