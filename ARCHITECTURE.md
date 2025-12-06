# Recurrent Attention Model (RAM) Architecture

## Overview
The RAM processes images by adaptively selecting a sequence of regions to attend to, using a recurrent attention mechanism.

## Architecture Flow

```
Input Image (H, W)
    ↓
[Glimpse Sensor] → Glimpse Patches
    ↓
[Glimpse Network] → Glimpse Feature Vector
    ↓
[Core Network] → Hidden State
    ↓
    ├─→ [Location Network] → Next Location (for t < num_glimpses-1)
    ├─→ [Value Network] → Value Estimate
    └─→ [Action Network] → Classification (at t = num_glimpses-1)
```

## Component Details

### Default Hyperparameters
- `glimpse_size = 8` (8×8 patches)
- `num_scales = 1` (single resolution scale)
- `num_glimpses = 6` (6 glimpses per image)
- `hidden_size = 256` (hidden state dimension)
- `num_actions = 10` (MNIST: 10 classes)
- `location_std = 0.2` (location policy variance)
- `glimpse_hidden_size = 128` (glimpse network hidden dimension)

---

## 1. Glimpse Sensor

**Purpose**: Extracts retina-like multi-scale patches from image around a location

**Input**:
- `image`: (H, W) - Single grayscale image (e.g., 28×28 for MNIST)
- `location`: (2,) - (x, y) coordinates in range [-1, 1]

**Process**:
- For each scale `s` in `[0, num_scales)`:
  - Extract patch of size `glimpse_size × (2^s)` centered at location
  - Resize to `glimpse_size × glimpse_size`
- Concatenate all patches

**Output**:
- `glimpse_patch`: (num_scales × glimpse_size × glimpse_size,)
  - With defaults: (1 × 8 × 8,) = (64,)

---

## 2. Glimpse Network

**Purpose**: Encodes glimpse patches and location into a feature vector

**Architecture**:
```
Glimpse Encoder:
  Input: (64,) → Linear(64, 128) → ReLU → (128,)

Location Encoder:
  Input: (2,) → Linear(2, 128) → ReLU → (128,)

Combine:
  Concat[(128,), (128,)] → (256,) → Linear(256, 256) → ReLU → (256,)
```

**Input**:
- `glimpse_patch`: (64,) - Flattened glimpse patches
- `location`: (2,) - (x, y) coordinates

**Output**:
- `glimpse_feature`: (256,) - Combined feature vector

---

## 3. Core Network

**Purpose**: Maintains recurrent hidden state across glimpses

**Architecture** (Simple RNN mode, default):
```
h_t = ReLU(Linear(h_{t-1}) + Linear(g_t))

Where:
  - h_{t-1}: (256,) - Previous hidden state
  - g_t: (256,) - Current glimpse feature
  - h_t: (256,) - New hidden state
```

**Input**:
- `glimpse_feature`: (256,) - From glimpse network
- `hidden_state`: (256,) or None - Previous hidden state (initialized to zeros)

**Output**:
- `hidden_state`: (256,) - Updated hidden state

**Alternative**: LSTM mode (when `use_lstm=True`)
- Uses `nn.LSTM(256, 256)` with hidden and cell states

---

## 4. Location Network

**Purpose**: Predicts next location to attend to (used for t < num_glimpses-1)

**Architecture**:
```
Input: (256,) → Linear(256, 2) → Tanh → (2,)
Sample: Normal(mean, std=0.2) → Clamp to [-1, 1]
```

**Input**:
- `hidden_state`: (256,) - Current hidden state

**Output**:
- `location`: (2,) - Next (x, y) location in [-1, 1]
- `log_prob`: scalar - Log probability of sampled location

---

## 5. Value Network

**Purpose**: Estimates expected cumulative reward (baseline for REINFORCE)

**Architecture**:
```
Input: (256,) → Linear(256, 1) → (1,)
```

**Input**:
- `hidden_state`: (256,) - Current hidden state

**Output**:
- `value`: scalar - Estimated value

---

## 6. Action Network

**Purpose**: Classifies the image (used at t = num_glimpses-1)

**Architecture**:
```
Input: (256,) → Linear(256, 10) → Softmax → (10,)
```

**Input**:
- `hidden_state`: (256,) - Final hidden state

**Output**:
- `logits`: (10,) - Class logits
- `probs`: (10,) - Class probabilities
- `action`: scalar - Sampled class (0-9)
- `log_prob`: scalar - Log probability of action

---

## Complete Forward Pass (num_glimpses=6)

### Step 0 (Initial):
- **Location**: Random in [-1, 1] → (2,)
- **Glimpse Sensor**: Extract patch → (64,)
- **Glimpse Network**: Encode → (256,)
- **Core Network**: Update state → (256,)
- **Value Network**: Estimate value → scalar

### Steps 1-4 (Intermediate):
- **Location Network**: Sample next location → (2,)
- **Glimpse Sensor**: Extract patch → (64,)
- **Glimpse Network**: Encode → (256,)
- **Core Network**: Update state → (256,)
- **Value Network**: Estimate value → scalar

### Step 5 (Final):
- **Glimpse Sensor**: Extract patch → (64,)
- **Glimpse Network**: Encode → (256,)
- **Core Network**: Update state → (256,)
- **Value Network**: Estimate value → scalar
- **Action Network**: Classify → (10,) → action (0-9)

---

## Tensor Shape Summary

| Component | Input Shape | Output Shape |
|-----------|-------------|--------------|
| **Input Image** | (H, W) | - |
| **Glimpse Sensor** | (H, W), (2,) | (64,) |
| **Glimpse Network** | (64,), (2,) | (256,) |
| **Core Network** | (256,), (256,) | (256,) |
| **Location Network** | (256,) | (2,), scalar |
| **Value Network** | (256,) | scalar |
| **Action Network** | (256,) | (10,), scalar |

---

## Model Output Structure

The `forward()` method returns a dictionary:

```python
{
    "actions": (1,) - Final classification action
    "locations": (5,) - Sampled locations (num_glimpses-1)
    "location_log_probs": (5,) - Log probs for each location
    "action_log_probs": (1,) - Log prob of final action
    "values": (6,) - Value estimates for each glimpse
    "hidden_state": (256,) - Final hidden state
}
```

---

## Parameter Counts (Approximate)

- **Glimpse Network**:
  - Glimpse encoder: 64×128 + 128 = 8,320
  - Location encoder: 2×128 + 128 = 384
  - Combine: 256×256 + 256 = 65,792
  - **Total**: ~74,496

- **Core Network** (Simple RNN):
  - Hidden linear: 256×256 + 256 = 65,792
  - Input linear: 256×256 + 256 = 65,792
  - **Total**: ~131,584

- **Location Network**:
  - Location linear: 256×2 + 2 = 514
  - **Total**: ~514

- **Value Network**:
  - Value linear: 256×1 + 1 = 257
  - **Total**: ~257

- **Action Network**:
  - Action linear: 256×10 + 10 = 2,570
  - **Total**: ~2,570

**Total Model Parameters**: ~209,421


