import torch
import torch.nn as nn

class MLPThicknessModel(nn.Module):
    """MLP model for regressing Acetal and Air gap thickness from 200-point S11 data.
    Adapted from Islam et al. (2024) MLP approach for dielectric layer regression
    Applies feature normalization (LayerNorm) to input as recommended in mmWave pipeline
    """
    def __init__(self, input_features: int = 400, hidden_sizes: list = [128, 64, 32]):
        super().__init__()
        # Normalise input features (2x200) to zero-mean, unit-variance
        self.norm = nn.LayerNorm(input_features)
        # Define MLP layers
        layers = []
        in_size = input_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU(inplace=True))
            # Using dropout to improve generalisation (common in similar works)
            layers.append(nn.Dropout(0.1))
            in_size = h
        layers.append(nn.Linear(in_size, 2))  # output layer for 2 thickness values
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input x of shape (batch, 2, 200)
        batch_size = x.size(0)
        # Flatten magnitude and phase features into a single vector per sample
        x = x.view(batch_size, -1)  # shape (batch, 400)
        # Normalise features across the 400-length vector
        x = self.norm(x)
        # Forward through MLP
        out = self.mlp(x)
        return out

# Example usage:
# model = MLPThicknessModel()
# pred = model(torch.randn(16, 2, 200))  # 16 samples, each with 2x200 features
