import torch.nn as nn
from DataPipeline.FeatureExtraction import N_MELS, N_FFT

STFT_BINS = N_FFT // 2 + 1   # 257

class BasicMLP(nn.Module):
    """Single-frame MLP baseline — no temporal context."""

    def __init__(self, n_mels=N_MELS, output_dim=STFT_BINS):
        super().__init__()
        self.n_mels = n_mels
        self.net = nn.Sequential(
            nn.Linear(n_mels, 256), nn.ReLU(),
            nn.Linear(256, 256),    nn.ReLU(),
            nn.Linear(256, output_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, window_size * n_mels) → extract center frame only
        c = x.shape[1] // (2 * self.n_mels)
        frame = x[:, c * self.n_mels:(c + 1) * self.n_mels]
        return self.net(frame)

if __name__ == "__main__":
    import torch
    model = BasicMLP()
    x = torch.randn(8, 320)   # batch=8, window=5 × mels=64
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {params:,}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
