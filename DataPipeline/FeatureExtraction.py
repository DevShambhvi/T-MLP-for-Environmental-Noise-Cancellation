import numpy as np
import librosa

# ── Shared feature-extraction constants ─────────────────────────────────────
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64

def compute_stft(audio):
    """
    Compute Short-Time Fourier Transform (STFT).
    """
    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    return magnitude, phase

def compute_mel_spectrogram(audio):
    """
    Compute the Mel Spectrogram.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    return mel

def compute_log_mel(audio):
    """
    Compute Log-Mel Spectrogram
    """
    mel = compute_mel_spectrogram(audio)

    log_mel = librosa.power_to_db(
        mel,
        ref = np.max
    )
    return log_mel


def normalize_features(features):
    """
    Apply per-utterance mean/variance normalization.

    Args:
        features (np.ndarray): Feature matrix (e.g. log-Mel spectrogram).

    Returns:
        np.ndarray: Zero-mean, unit-variance normalized features.
    """
    mean = np.mean(features)
    std = np.std(features)
    if std < 1e-10:
        return features - mean
    return (features - mean) / std

def create_sliding_window(features, window_size = 5):
    """
    Create overlapping sliding windows from feature frames.

    Args:
        features (np.ndarray): Normalized Log-Mel spectrogram
                               of shape (n_mels, time_frames).
        window_size (int): Number of consecutive frames.

    Returns:
        np.ndarray: Sliding windows of shape
                    (num_windows, window_size * n_mels)
    """
    features = features.T 
    windows = []

    for i in range(len(features) - window_size + 1):
        window = features[i:i + window_size]
        windows.append(window.flatten())

    return np.array(windows)