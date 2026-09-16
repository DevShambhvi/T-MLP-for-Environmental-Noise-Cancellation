from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16000

def load_audio(audio_path):

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    return audio, sample_rate

def convert_to_mono(audio):

    if audio.ndim == 1:
        return audio

    mono_audio = np.mean(
        audio,
        axis=1
    )

    return mono_audio

def resample_audio(audio, original_sample_rate):

    if original_sample_rate == TARGET_SAMPLE_RATE:
        return audio

    resampled_audio = resample_poly(
        audio,
        TARGET_SAMPLE_RATE,
        original_sample_rate
    )

    return resampled_audio

def validate_audio(audio):

    if len(audio) == 0:
        raise ValueError("Audio waveform is empty.")

    if not np.isfinite(audio).all():
        raise ValueError(
            "Audio waveform contains NaN or Infinity values."
        )

    return audio

def preprocess_audio(audio_path):

    audio, original_sample_rate = load_audio(audio_path)

    audio = convert_to_mono(audio)

    audio = validate_audio(audio)

    audio = resample_audio(
    audio,
    original_sample_rate
)

    return audio, TARGET_SAMPLE_RATE

def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    speech_path = PROJECT_ROOT / "data" / "speech"
    noise_path = PROJECT_ROOT / "data" / "noise"

    speech_files = [
        file
        for file in speech_path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in {".wav", ".flac"}
    ]

    noise_files = [
        file
        for file in noise_path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in {".wav", ".flac"}
    ]

    if speech_files:
        speech_audio, speech_sr = preprocess_audio(
            speech_files[0]
        )

        print("SPEECH")
        print(f"File: {speech_files[0].name}")
        print(f"Processed sample rate: {speech_sr}")
        print(f"Processed samples: {len(speech_audio)}")

    if noise_files:
        noise_audio, noise_sr = preprocess_audio(
            noise_files[0]
        )

        print("\nNOISE")
        print(f"File: {noise_files[0].name}")
        print(f"Processed sample rate: {noise_sr}")
        print(f"Processed samples: {len(noise_audio)}")


if __name__ == "__main__":
    main()