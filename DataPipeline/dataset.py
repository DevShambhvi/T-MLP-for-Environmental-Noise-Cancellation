import os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from DataPipeline.datapipeline import load_audio, match_length, mix_audio
from DataPipeline.FeatureExtraction import (
    compute_stft, compute_log_mel, normalize_features, create_sliding_window,
)

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")

def _list_audio(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(AUDIO_EXT))

def split_files(directory, ratios=(0.8, 0.1), seed=42):
    files = _list_audio(directory)
    random.seed(seed)
    random.shuffle(files)
    t, v = int(len(files) * ratios[0]), int(len(files) * sum(ratios))
    return files[:t], files[t:v], files[v:]

def compute_irm(speech, noise):
    s, _ = compute_stft(speech)
    n, _ = compute_stft(noise)
    return np.sqrt(s**2 / (s**2 + n**2 + 1e-10))

class NoiseCancellationDataset(Dataset):
    def __init__(self, speech_files, noise_files, win=5, snr=(-5, 20)):
        self.speech_files, self.noise_files = speech_files, noise_files
        self.win, self.snr = win, snr

    def __len__(self): return len(self.speech_files)

    def __getitem__(self, i):
        speech, _ = load_audio(self.speech_files[i])
        noise, _  = load_audio(random.choice(self.noise_files))
        noise  = match_length(speech, noise)
        noisy  = mix_audio(speech, noise, random.uniform(*self.snr))
        irm    = compute_irm(speech, noise)
        wins   = create_sliding_window(normalize_features(compute_log_mel(noisy)), self.win)
        return torch.tensor(wins, dtype=torch.float32), \
               torch.tensor(irm.T[:len(wins)], dtype=torch.float32)

def collate_fn(batch):
    return torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

def create_dataloaders(speech_dir, noise_dir, batch_size=64, win=5, workers=0):
    splits = split_files(speech_dir)
    noise  = _list_audio(noise_dir)
    loaders = tuple(
        DataLoader(NoiseCancellationDataset(s, noise, win), batch_size=batch_size,
                   shuffle=sh, collate_fn=collate_fn, num_workers=workers)
        for s, sh in zip(splits, (True, False, False))
    )
    print(f"Train: {len(splits[0])} | Val: {len(splits[1])} | Test: {len(splits[2])}")
    return loaders

if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    train, val, test = create_dataloaders(
        os.path.join(ROOT, "data", "speech"),
        os.path.join(ROOT, "data", "noise"), batch_size=4,
    )
    x, y = next(iter(train))
    print(f"Input:  {x.shape}  [{x.min():.3f}, {x.max():.3f}]")
    print(f"Target: {y.shape} [{y.min():.3f}, {y.max():.3f}]")
