import numpy as np
import torch
import librosa
from DataPipeline.FeatureExtraction import N_FFT, HOP_LENGTH

def compute_nmse(pred, target):
    """Normalized Mean Squared Error (Linear & dB)."""
    mse = np.sum((pred - target) ** 2)
    norm = np.sum(target ** 2) + 1e-10
    nmse_linear = mse / norm
    nmse_db = 10 * np.log10(nmse_linear + 1e-10)
    return nmse_linear, nmse_db

def compute_f1_precision_recall(pred, target, threshold=0.5):
    """Binary mask classification metrics at threshold 0.5."""
    pred_bin = (pred >= threshold).astype(int)
    target_bin = (target >= threshold).astype(int)

    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1

def compute_sdr(clean_audio, enhanced_audio):
    """Signal-to-Distortion Ratio (dB)."""
    min_len = min(len(clean_audio), len(enhanced_audio))
    s = clean_audio[:min_len]
    e = enhanced_audio[:min_len]
    noise = s - e
    s_power = np.sum(s ** 2)
    n_power = np.sum(noise ** 2) + 1e-10
    return 10 * np.log10(s_power / n_power)

def compute_stoi_safe(clean_audio, enhanced_audio, sr=16000):
    """Short-Time Objective Intelligibility (STOI) score."""
    try:
        from pystoi import stoi  # type: ignore
        min_len = min(len(clean_audio), len(enhanced_audio))
        return stoi(clean_audio[:min_len], enhanced_audio[:min_len], sr, extended=False)
    except ImportError:
        # Fallback spectral correlation metric if pystoi is not installed
        min_len = min(len(clean_audio), len(enhanced_audio))
        s_stft = np.abs(librosa.stft(clean_audio[:min_len], n_fft=N_FFT, hop_length=HOP_LENGTH))
        e_stft = np.abs(librosa.stft(enhanced_audio[:min_len], n_fft=N_FFT, hop_length=HOP_LENGTH))
        corr = np.corrcoef(s_stft.flatten(), e_stft.flatten())[0, 1]
        return float(np.clip(corr, 0.0, 1.0))

def estimate_hardware_efficiency(model_type, param_count, is_int8=False):
    """
    Analytical MCU memory and MAC estimation.
    Bytes per weight: FP32 = 4 bytes, INT8 = 1 byte.
    """
    bytes_per_param = 1 if is_int8 else 4
    weights_ram_kb = (param_count * bytes_per_param) / 1024.0

    if model_type in ['mlp', 't_mlp', 't_mlp_qat']:
        # Dense layers MAC count per frame
        # T-MLP: 320x256 + 256x128 + 128x257 = 81,920 + 32,768 + 32,896 = 147,584 MACs
        # Basic MLP: 64x256 + 256x256 + 256x257 = 16,384 + 65,536 + 65,792 = 147,712 MACs
        macs_per_frame = param_count  # linear layer weight count equals MAC count
        act_buffer_kb = (320 + 256 + 128 + 257) * 4 / 1024.0
    elif model_type == 'gru':
        # GRU: 3*(64*160 + 160*160 + 160) = 108,000 MACs per frame + head 160x257 = 41,120 MACs
        macs_per_frame = param_count
        act_buffer_kb = (64 + 160*3 + 257) * 4 / 1024.0
    else:
        macs_per_frame = param_count
        act_buffer_kb = 2.0

    total_ram_kb = weights_ram_kb + act_buffer_kb
    macs_per_sec = macs_per_frame * (16000 / HOP_LENGTH)  # 62.5 frames/sec

    return {
        "macs_per_frame": int(macs_per_frame),
        "macs_per_sec": int(macs_per_sec),
        "weights_ram_kb": round(weights_ram_kb, 2),
        "total_ram_kb": round(total_ram_kb, 2)
    }
