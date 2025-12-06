# Recurrent Attention Model for Image Classification

Implementation of the Recurrent Attention Model for Image Classification


## Overview

This repository contains a PyTorch implementation of the Recurrent Attention Model, which processes images by adaptively selecting a sequence of regions to attend to. The model uses:

- **Glimpse Sensor**: Extracts retina-like multi-resolution patches from images
- **Glimpse Network**: Processes glimpse patches and location coordinates
- **Core Network**: RNN that maintains internal state (simple RNN for classification)
- **Location Network**: Outputs next location to attend to (trained with REINFORCE)
- **Action Network**: Outputs classification/action decisions (trained with supervised learning for classification)
- **Value Network**: Estimates expected cumulative reward for variance reduction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Recurrent-Modeling_Of_Visual-Attention_for_ImageClassification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── models/              # Core model components
│   ├── glimpse_sensor.py      # Retina-like glimpse extraction
│   ├── glimpse_network.py      # Glimpse processing network
│   ├── core_network.py         # RNN core (simple RNN or LSTM)
│   ├── location_network.py     # Location policy network
│   ├── action_network.py       # Action/classification network
│   ├── value_network.py        # Value function (baseline)
│   └── ram.py                  # Main RAM model
├── utils/               # Utilities
│   ├── datasets.py             # MNIST variant datasets
│   └── trainer.py              # Training utilities
├── scripts/             # Training and evaluation scripts
│   ├── train_mnist.py          # Train on MNIST variants
│   └── evaluate.py             # Evaluate trained models
└── data/                # Data directory (created automatically)
```

## Usage

### Training on MNIST

Train on standard MNIST:
```bash
python scripts/train_mnist.py --dataset mnist --num_glimpses 6 --num_epochs 10
```

### Model Checkpoints

During training, the model automatically saves checkpoints to `./checkpoints/` (or custom directory via `--save_dir`):
- `{dataset}_best.pth`: Best performing model (highest test accuracy)
- `{dataset}_latest.pth`: Most recent epoch

Each checkpoint contains:
- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Training configuration
- Epoch number and test accuracy

Custom save directory:
```bash
python scripts/train_mnist.py --save_dir ./my_models
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --model_path ./checkpoints/mnist_best.pth --dataset mnist
```
