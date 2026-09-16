import numpy as np
from pathlib import Path
from data_preprocessing import preprocess_audio

def calculate_power(audio):

    power = np.mean(audio ** 2)

    return power

def scale_noise_to_snr(speech, noise, target_snr_db):

    speech_power = calculate_power(speech)
    noise_power = calculate_power(noise)

    if noise_power == 0:
        raise ValueError("Noise power is zero. Cannot scale noise.")

    target_noise_power = (
        speech_power /
        (10 ** (target_snr_db / 10))
    )

    scaling_factor = np.sqrt(
        target_noise_power / noise_power
    )

    scaled_noise = noise * scaling_factor

    return scaled_noise

def calculate_snr(speech, noise):

    speech_power = calculate_power(speech)
    noise_power = calculate_power(noise)

    if noise_power == 0:
        raise ValueError("Noise power is zero. SNR is undefined.")

    snr_db = 10 * np.log10(
        speech_power / noise_power
    )

    return snr_db

def mix_audio(speech, noise, target_snr_db):

    scaled_noise = scale_noise_to_snr(
        speech,
        noise,
        target_snr_db
    )

    noisy_speech = speech + scaled_noise

    return noisy_speech, scaled_noise

def match_audio_length(speech, noise, rng):

    speech_length = len(speech)
    noise_length = len(noise)

    if noise_length >= speech_length:

        max_start = noise_length - speech_length

        start = rng.integers(
            0,
            max_start + 1
        )

        matched_noise = noise[
            start:start + speech_length
        ]

        return matched_noise

    repeats = int(
        np.ceil(speech_length / noise_length)
    )

    repeated_noise = np.tile(
        noise,
        repeats
    )

    matched_noise = repeated_noise[
        :speech_length
    ]

    return matched_noise

def create_noisy_sample(
    speech,
    noise,
    target_snr_db,
    rng
):

    matched_noise = match_audio_length(
        speech,
        noise,
        rng
    )

    noisy_speech, scaled_noise = mix_audio(
        speech,
        matched_noise,
        target_snr_db
    )

    return noisy_speech, speech, scaled_noise


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    speech_path = PROJECT_ROOT / "data" / "speech"
    noise_path = PROJECT_ROOT / "data" / "noise"

    speech_files = list(
        speech_path.rglob("*.flac")
    )

    noise_files = list(
        noise_path.rglob("*.wav")
    )

    rng = np.random.default_rng(42)

    speech, speech_sr = preprocess_audio(
        speech_files[0]
    )

    noise, noise_sr = preprocess_audio(
        noise_files[0]
    )

    target_snr_db = 10

    noisy_speech, clean_speech, scaled_noise = (
        create_noisy_sample(
            speech,
            noise,
            target_snr_db,
            rng
        )
    )

    actual_snr = calculate_snr(
        clean_speech,
        scaled_noise
    )

    print("MIXING TEST")
    print(f"Speech samples: {len(clean_speech)}")
    print(f"Noise samples: {len(scaled_noise)}")
    print(f"Noisy speech samples: {len(noisy_speech)}")
    print(f"Target SNR: {target_snr_db} dB")
    print(f"Actual SNR: {actual_snr:.2f} dB")