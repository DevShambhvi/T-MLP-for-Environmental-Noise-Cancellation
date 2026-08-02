import torch.nn as nn
from DataPipeline.FeatureExtraction import N_MELS, N_FFT

STFT_BINS = N_FFT // 2 + 1   # 257

class TMLP(nn.Module):
    """Temporal MLP — sliding window of stacked mel frames."""

    def __init__(self, window_size=5, n_mels=N_MELS, output_dim=STFT_BINS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size * n_mels, 256), nn.ReLU(),
            nn.Linear(256, 128),                  nn.ReLU(),
            nn.Linear(128, output_dim),           nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    import torch
    model = TMLP()
    x = torch.randn(8, 320)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {params:,}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
