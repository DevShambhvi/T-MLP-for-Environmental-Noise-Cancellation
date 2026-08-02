import os
import random 

import librosa
import numpy as np

# ── Directory paths (relative to this file's location) ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEECH_DIR = os.path.join(SCRIPT_DIR, "..", "data", "speech")
NOISE_DIR  = os.path.join(SCRIPT_DIR, "..", "data", "noise")


def load_audio(file_path, sr=16000):
    """
    Load an audio file.
    returns:
    audio (np.ndarray) : Audio Waveforms
    sample_rate (int) : Sampling rate
    """
    audio, sample_rate = librosa.load(
        file_path,
        sr=sr,
        mono=True
    )
    return audio, sample_rate


def match_length(speech, noise):
    """
    Adjust the noise length to match the speech length.
    speech(np.ndarray): Clean waveforms
    noise(np.ndarray): Noise waveforms    
    """
    speech_len = len(speech)
    noise_len = len(noise)
    if (noise_len > speech_len):
        start = random.randint(0, noise_len - speech_len)
        noise = noise[start: start + speech_len]
    elif (noise_len < speech_len):
        repeats = int(np.ceil(speech_len/noise_len))
        noise = np.tile(noise, repeats)
        noise = noise[:speech_len]
    return noise

def mix_audio(speech, noise, snr_db):
    """
    Mix speech and noise at a desired SNR.
    speech (np.ndarray) : clean waverform
    noise (np.ndarray) : noise waveform
    snr_db (float): Desired Signal-to-noise ratio in dB
    """
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return speech.copy()
    scale = np.sqrt(
        speech_power / (noise_power * (10 ** (snr_db / 10)))
    )
    scaled_noise = noise * scale
    noisy_speech = scaled_noise + speech
    return noisy_speech

def generate_sample(speech_dir, noise_dir):
    """
    Generate one noisy speech sample.
    """
    if not os.path.isdir(speech_dir):
        raise FileNotFoundError(f"Speech directory not found: {speech_dir}")
    if not os.path.isdir(noise_dir):
        raise FileNotFoundError(f"Noise directory not found: {noise_dir}")

    audio_ext = (".wav", ".flac", ".mp3", ".ogg")
    speech_files = [f for f in os.listdir(speech_dir) if f.lower().endswith(audio_ext)]
    noise_files = [f for f in os.listdir(noise_dir) if f.lower().endswith(audio_ext)]
    speech_file = random.choice(speech_files)
    noise_file = random.choice(noise_files)
    speech, _ = load_audio(os.path.join(speech_dir, speech_file))
    noise, _ = load_audio(os.path.join(noise_dir, noise_file))
    noise = match_length(speech, noise)
    snr = random.uniform(-5, 20)
    noisy = mix_audio(speech, noise, snr)
    return speech, noise, noisy, snr

# ── Main: quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Speech dir: {os.path.abspath(SPEECH_DIR)}")
    print(f"Noise dir:  {os.path.abspath(NOISE_DIR)}")
    print(f"Speech files found: {len(os.listdir(SPEECH_DIR))}")
    print(f"Noise files found:  {len(os.listdir(NOISE_DIR))}")

    speech, noise, noisy, snr = generate_sample(SPEECH_DIR, NOISE_DIR)
    print(f"SNR: {snr:.1f} dB")
    print(f"Speech shape: {speech.shape}, Noise shape: {noise.shape}, Noisy shape: {noisy.shape}")


