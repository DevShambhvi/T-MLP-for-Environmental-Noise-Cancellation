import torch
import torch.nn as nn
from DataPipeline.FeatureExtraction import N_MELS, N_FFT

STFT_BINS = N_FFT // 2 + 1

class GRUBaseline(nn.Module):
    """GRU baseline — processes mel frames as a sequence."""

    def __init__(self, n_mels=N_MELS, hidden=160, output_dim=STFT_BINS):
        super().__init__()
        self.n_mels = n_mels
        self.gru = nn.GRU(n_mels, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, output_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, window_size * n_mels) → (batch, window_size, n_mels)
        seq = x.view(x.size(0), -1, self.n_mels)
        _, h = self.gru(seq)              # h: (1, batch, hidden)
        return self.head(h.squeeze(0))    # (batch, output_dim)

if __name__ == "__main__":
    model = GRUBaseline()
    x = torch.randn(8, 320)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {params:,}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
