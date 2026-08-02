import torch
import torch.nn as nn
import torch.ao.quantization as quant
from DataPipeline.FeatureExtraction import N_MELS, N_FFT

STFT_BINS = N_FFT // 2 + 1

class TMLP_QAT(nn.Module):
    """T-MLP with fake-quantization for Quantization-Aware Training."""

    def __init__(self, window_size=5, n_mels=N_MELS, output_dim=STFT_BINS):
        super().__init__()
        self.quant   = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        self.net = nn.Sequential(
            nn.Linear(window_size * n_mels, 256), nn.ReLU(),
            nn.Linear(256, 128),                  nn.ReLU(),
            nn.Linear(128, output_dim),           nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.net(x)
        x = self.dequant(x)
        return x

    def fuse_and_prepare(self, backend="x86"):
        """Call before QAT fine-tuning to insert fake-quant observers."""
        self.qconfig = quant.get_default_qat_qconfig(backend)
        quant.prepare_qat(self, inplace=True)
        return self

    def convert_to_int8(self):
        """Call after QAT training to produce the final quantized model."""
        quant.convert(self, inplace=True)
        return self

if __name__ == "__main__":
    model = TMLP_QAT()
    x = torch.randn(8, 320)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {params:,}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Show QAT prep works
    model.fuse_and_prepare()
    y_qat = model(x)
    print(f"After QAT prep — output: {y_qat.shape}")
