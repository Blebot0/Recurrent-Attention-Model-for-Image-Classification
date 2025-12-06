import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreNetwork(nn.Module):
    """
    Core RNN network that maintains internal state.
    For classification: simple RNN with rectifier units.
    For dynamic environments: LSTM units.
    """

    def __init__(self, input_size, hidden_size, use_lstm=False):
        """
        Initialize the core network.

        Args:
            input_size: Size of input (glimpse feature vector)
            hidden_size: Size of hidden state
            use_lstm: If True, use LSTM; otherwise use simple RNN
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.use_lstm = use_lstm

        if use_lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            # Simple RNN: ht = ReLU(Linear(ht-1) + Linear(gt))
            self.hidden_linear = nn.Linear(hidden_size, hidden_size)
            self.input_linear = nn.Linear(input_size, hidden_size)

    def forward(self, glimpse_feature, hidden_state=None):
        """
        Forward pass through the core network.

        Args:
            glimpse_feature: Glimpse feature vector from glimpse network
            hidden_state: Previous hidden state (ht-1)

        Returns:
            New hidden state (ht)
        """
        if self.use_lstm:
            # LSTM expects (batch, seq_len, input_size)
            if len(glimpse_feature.shape) == 1:
                glimpse_feature = glimpse_feature.unsqueeze(0).unsqueeze(0)
            elif len(glimpse_feature.shape) == 2:
                glimpse_feature = glimpse_feature.unsqueeze(1)

            if hidden_state is None:
                h0 = torch.zeros(
                    1, glimpse_feature.shape[0], self.hidden_size
                ).to(glimpse_feature.device)
                c0 = torch.zeros(
                    1, glimpse_feature.shape[0], self.hidden_size
                ).to(glimpse_feature.device)
                hidden_state = (h0, c0)

            output, new_hidden = self.rnn(glimpse_feature, hidden_state)
            return output.squeeze(1), new_hidden
        else:
            # Simple RNN: ht = ReLU(Linear(ht-1) + Linear(gt))
            if hidden_state is None:
                hidden_state = torch.zeros(
                    glimpse_feature.shape[0] if len(glimpse_feature.shape) > 1 else 1,
                    self.hidden_size,
                ).to(glimpse_feature.device)

            if len(glimpse_feature.shape) == 1:
                glimpse_feature = glimpse_feature.unsqueeze(0)
            if len(hidden_state.shape) == 1:
                hidden_state = hidden_state.unsqueeze(0)

            h_hidden = self.hidden_linear(hidden_state)
            h_input = self.input_linear(glimpse_feature)
            new_hidden = F.relu(h_hidden + h_input)

            return new_hidden.squeeze(0) if new_hidden.shape[0] == 1 else new_hidden

