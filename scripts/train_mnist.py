import argparse
import sys
from pathlib import Path
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import RecurrentAttentionModel
from utils.datasets import (
    get_mnist_datasets,
)
from utils.trainer import RAMTrainer


def setup_logger(log_dir, experiment_name):
    """
    Setup advanced logging system with both file and console output.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment for the log file
        
    Returns:
        logger: Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('RAM_Training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler - detailed logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_path / f"{experiment_name}_{timestamp}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - simple output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


def plot_training_curves(history, save_path):
    """
    Plot and save training curves.
    
    Args:
        history: Dictionary containing training metrics history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    if 'train_loss' in history and len(history['train_loss']) > 0:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss (Per Batch)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    if 'train_accuracy' in history and len(history['train_accuracy']) > 0:
        axes[0, 1].plot(history['train_accuracy'], label='Train Acc', 
                       color='green', linewidth=2)
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training Accuracy (Per Batch)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epoch-level Test Accuracy
    if 'test_accuracy' in history and len(history['test_accuracy']) > 0:
        axes[1, 0].plot(history['test_accuracy'], marker='o', 
                       label='Test Acc', color='red', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Test Accuracy (Per Epoch)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss Components (if available)
    if 'action_loss' in history and len(history['action_loss']) > 0:
        axes[1, 1].plot(history['action_loss'], label='Action Loss', alpha=0.7)
        if 'location_loss' in history and len(history['location_loss']) > 0:
            axes[1, 1].plot(history['location_loss'], label='Location Loss', alpha=0.7)
        if 'value_loss' in history and len(history['value_loss']) > 0:
            axes[1, 1].plot(history['value_loss'], label='Value Loss', alpha=0.7)
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save a simple loss curve
    plt.figure(figsize=(10, 6))
    if 'train_loss' in history and len(history['train_loss']) > 0:
        # Smooth the loss curve using moving average
        window_size = min(100, len(history['train_loss']) // 10 + 1)
        if len(history['train_loss']) >= window_size:
            smoothed_loss = np.convolve(
                history['train_loss'], 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            plt.plot(smoothed_loss, linewidth=2, label=f'Smoothed (window={window_size})')
        plt.plot(history['train_loss'], alpha=0.3, label='Raw')
        plt.xlabel('Batch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = save_path.parent / f"{save_path.stem}_loss_only.png"
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()


def train_mnist(
    dataset_type="mnist",
    image_size=28,
    num_glimpses=6,
    glimpse_size=8,
    num_scales=1,
    hidden_size=256,
    num_epochs=10,
    batch_size=64,
    lr=1e-3,
    momentum=0.9,
    device="cpu",
    data_dir="./data",
    save_dir="./checkpoints",
    log_dir="./logs",
    plot_dir="./plots",
):
    """
    Train RAM model on MNIST with advanced logging and visualization.

    Args:
        dataset_type: Type of dataset (mnist)
        image_size: Size of images
        num_glimpses: Number of glimpses
        glimpse_size: Size of glimpse patches
        num_scales: Number of resolution scales
        hidden_size: Hidden state size
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        momentum: Momentum for SGD
        device: Device to use
        data_dir: Data directory
        save_dir: Directory to save model checkpoints
        log_dir: Directory to save log files
        plot_dir: Directory to save training plots
    """
    # Create directories
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    experiment_name = f"{dataset_type}_g{num_glimpses}_h{hidden_size}"
    logger = setup_logger(log_dir, experiment_name)
    
    # Log experiment configuration
    logger.info("=" * 80)
    logger.info("RECURRENT ATTENTION MODEL - TRAINING SESSION")
    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Dataset: {dataset_type}")
    logger.info(f"  Image Size: {image_size}x{image_size}")
    logger.info(f"  Num Glimpses: {num_glimpses}")
    logger.info(f"  Glimpse Size: {glimpse_size}")
    logger.info(f"  Num Scales: {num_scales}")
    logger.info(f"  Hidden Size: {hidden_size}")
    logger.info(f"  Num Epochs: {num_epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Learning Rate: {lr}")
    logger.info(f"  Momentum: {momentum}")
    logger.info(f"  Device: {device}")
    logger.info("")
    
    # Load dataset
    logger.info("Loading datasets...")
    if dataset_type == "mnist":
        train_dataset, test_dataset = get_mnist_datasets(
            data_dir=data_dir, image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    logger.info("")

    # Create model
    logger.info("Initializing model...")
    model = RecurrentAttentionModel(
        glimpse_size=glimpse_size,
        num_scales=num_scales,
        num_glimpses=num_glimpses,
        hidden_size=hidden_size,
        num_actions=10,
        location_std=0.2,
        use_lstm=False,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {num_params:,}")
    logger.info(f"  Trainable parameters: {num_trainable:,}")
    logger.info("")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = RAMTrainer(
        model,
        lr=lr,
        momentum=momentum,
        device=device,
    )
    logger.info("  Optimizer: SGD")
    logger.info("")

    # Training history for plotting
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'action_loss': [],
        'location_loss': [],
        'value_loss': [],
        'test_accuracy': [],
        'best_accuracy': 0.0,
        'best_epoch': 0,
    }
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info("")
    
    best_accuracy = 0.0
    
    # Epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs", position=0, 
                      leave=True, ncols=100, colour='green')
    
    for epoch in epoch_pbar:
        epoch_losses = []
        epoch_accuracies = []
        epoch_action_losses = []
        epoch_location_losses = []
        epoch_value_losses = []

        # Batch progress bar
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                         position=1, leave=False, ncols=100, colour='blue')
        
        for batch_idx, (images, labels) in enumerate(batch_pbar):
            # Process batch (trainer handles single image at a time)
            losses = trainer.train_step_classification(
                images, labels, num_glimpses=num_glimpses
            )

            # Store losses for plotting
            history['train_loss'].append(losses["total_loss"])
            epoch_losses.append(losses["total_loss"])
            
            if "accuracy" in losses:
                history['train_accuracy'].append(losses["accuracy"])
                epoch_accuracies.append(losses["accuracy"])
            
            if "action_loss" in losses:
                history['action_loss'].append(losses["action_loss"])
                epoch_action_losses.append(losses["action_loss"])
            
            if "location_loss" in losses:
                history['location_loss'].append(losses["location_loss"])
                epoch_location_losses.append(losses["location_loss"])
            
            if "value_loss" in losses:
                history['value_loss'].append(losses["value_loss"])
                epoch_value_losses.append(losses["value_loss"])

            # Update progress bar with current metrics
            avg_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) > 0 else 0
            avg_acc = np.mean(epoch_accuracies[-100:]) if len(epoch_accuracies) > 0 else 0
            batch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.4f}'
            })

        # Close batch progress bar
        batch_pbar.close()
        
        # Compute epoch statistics
        avg_epoch_loss = np.mean(epoch_losses)
        avg_epoch_acc = np.mean(epoch_accuracies) if epoch_accuracies else 0.0
        
        # Evaluate on test set
        logger.info(f"Evaluating epoch {epoch+1}...")
        test_metrics = trainer.evaluate(test_loader, num_glimpses=num_glimpses)
        test_accuracy = test_metrics['accuracy']
        history['test_accuracy'].append(test_accuracy)
        
        # Log epoch results
        logger.info("")
        logger.info(f"Epoch {epoch+1}/{num_epochs} Summary:")
        logger.info(f"  Train Loss: {avg_epoch_loss:.6f}")
        logger.info(f"  Train Accuracy: {avg_epoch_acc:.4f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        if epoch_action_losses:
            logger.info(f"  Action Loss: {np.mean(epoch_action_losses):.6f}")
        if epoch_location_losses:
            logger.info(f"  Location Loss: {np.mean(epoch_location_losses):.6f}")
        if epoch_value_losses:
            logger.info(f"  Value Loss: {np.mean(epoch_value_losses):.6f}")
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'test_accuracy': test_accuracy,
            'train_accuracy': avg_epoch_acc,
            'train_loss': avg_epoch_loss,
            'history': history,
            'config': {
                'dataset_type': dataset_type,
                'image_size': image_size,
                'num_glimpses': num_glimpses,
                'glimpse_size': glimpse_size,
                'num_scales': num_scales,
                'hidden_size': hidden_size,
                'lr': lr,
                'momentum': momentum,
            }
        }
        
        # Save latest checkpoint
        latest_path = save_path / f"{dataset_type}_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best model (checkpoint for higher accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            history['best_accuracy'] = best_accuracy
            history['best_epoch'] = epoch + 1
            best_path = save_path / f"{dataset_type}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"  ✓ NEW BEST MODEL! Accuracy: {best_accuracy:.4f}")
            logger.info(f"    Saved to: {best_path}")
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'test_acc': f'{test_accuracy:.4f}',
            'best': f'{best_accuracy:.4f}'
        })
        
        # Plot training curves
        plot_file = plot_path / f"{experiment_name}_training_curves.png"
        plot_training_curves(history, plot_file)
        
        logger.info("")

    # Close epoch progress bar
    epoch_pbar.close()
    
    # Final summary
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Best Test Accuracy: {best_accuracy:.4f}")
    logger.info(f"Best Epoch: {history['best_epoch']}")
    logger.info(f"Total Epochs: {num_epochs}")
    logger.info("")
    logger.info("Saved Files:")
    logger.info(f"  Best Model: {save_path / f'{dataset_type}_best.pth'}")
    logger.info(f"  Latest Model: {save_path / f'{dataset_type}_latest.pth'}")
    logger.info(f"  Training Plots: {plot_path / f'{experiment_name}_training_curves.png'}")
    logger.info(f"  Loss Plot: {plot_path / f'{experiment_name}_training_curves_loss_only.png'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAM on MNIST")
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
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save training logs",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="./plots",
        help="Directory to save training plots",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("CUDA requested but not available. Using CPU instead.")
    else:
        device = "cpu"
        print("Using CPU.")

    train_mnist(
        dataset_type=args.dataset,
        image_size=args.image_size,
        num_glimpses=args.num_glimpses,
        glimpse_size=args.glimpse_size,
        num_scales=args.num_scales,
        hidden_size=args.hidden_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        device=device,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
    )

