
from pathlib import Path
import numpy as np

DATA_DIR = Path("processed_data")

clean_peaks = []
noise_peaks = []
mixture_peaks = []
high_peak_files = []

for split in ("train", "validation", "test"):
    for file in (DATA_DIR / split).glob("*.npz"):
        with np.load(file, allow_pickle=False) as data:
            clean = data["clean_speech"].astype(np.float64)
            noisy = data["noisy_speech"].astype(np.float64)
            noise = noisy - clean

        clean_peak = float(np.max(np.abs(clean)))
        noise_peak = float(np.max(np.abs(noise)))
        mixture_peak = float(np.max(np.abs(noisy)))

        clean_peaks.append(clean_peak)
        noise_peaks.append(noise_peak)
        mixture_peaks.append(mixture_peak)

        if mixture_peak > 1.0:
            high_peak_files.append(
                (file.name, clean_peak, noise_peak, mixture_peak)
            )

print("Samples checked:", len(mixture_peaks))
print("Clean speech maximum peak:", max(clean_peaks))
print("Scaled noise maximum peak:", max(noise_peaks))
print("Mixture maximum peak:", max(mixture_peaks))
print("Clean speech samples with peak > 1:",
      sum(x > 1.0 for x in clean_peaks))
print("Noise samples with peak > 1:",
      sum(x > 1.0 for x in noise_peaks))
print("Mixtures with peak > 1:", len(high_peak_files))
print("\nFirst 10 mixtures exceeding 1.0:")
for row in high_peak_files[:10]:
    print(row)
