
from pathlib import Path
import numpy as np

from data_preprocessing import preprocess_audio
from mixing import create_noisy_sample


SUPPORTED_EXTENSIONS = {".wav", ".flac"}


def get_audio_files(dataset_path):

    audio_files = [
        file
        for file in dataset_path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(audio_files)


def get_speaker_id(audio_path):

    filename_parts = audio_path.stem.split("-")

    speaker_id = filename_parts[0]

    return speaker_id


def split_dataset(
    audio_files,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    seed=42
):

    if not np.isclose(
        train_ratio
        + validation_ratio
        + test_ratio,
        1.0
    ):
        raise ValueError(
            "Dataset split ratios must add up to 1.0."
        )

    speakers = sorted(
        {
            get_speaker_id(audio_file)
            for audio_file in audio_files
        }
    )

    rng = np.random.default_rng(seed)

    shuffled_speakers = speakers.copy()

    rng.shuffle(shuffled_speakers)

    total_speakers = len(shuffled_speakers)

    train_end = int(
        total_speakers * train_ratio
    )

    validation_end = (
        train_end
        + int(total_speakers * validation_ratio)
    )

    train_speakers = set(
        shuffled_speakers[:train_end]
    )

    validation_speakers = set(
        shuffled_speakers[
            train_end:validation_end
        ]
    )

    test_speakers = set(
        shuffled_speakers[
            validation_end:
        ]
    )

    train_files = [
        file
        for file in audio_files
        if get_speaker_id(file) in train_speakers
    ]

    validation_files = [
        file
        for file in audio_files
        if get_speaker_id(file) in validation_speakers
    ]

    test_files = [
        file
        for file in audio_files
        if get_speaker_id(file) in test_speakers
    ]

    return (
        train_files,
        validation_files,
        test_files
    )


def select_noise_file(noise_files, rng):

    if not noise_files:
        raise ValueError(
            "Noise file list is empty."
        )

    noise_index = rng.integers(
        0,
        len(noise_files)
    )

    return noise_files[noise_index]


def select_speech_file(speech_files, rng):

    if not speech_files:
        raise ValueError(
            "Speech file list is empty."
        )

    speech_index = rng.integers(
        0,
        len(speech_files)
    )

    return speech_files[speech_index]


def generate_sample(
    speech_file,
    noise_file,
    target_snr_db,
    rng
):

    speech, _ = preprocess_audio(
        speech_file
    )

    noise, _ = preprocess_audio(
        noise_file
    )

    noisy_speech, clean_speech, scaled_noise = (
        create_noisy_sample(
            speech,
            noise,
            target_snr_db,
            rng
        )
    )

    return (
        noisy_speech,
        clean_speech,
        scaled_noise
    )


def save_sample(
    noisy_speech,
    clean_speech,
    speech_file,
    noise_file,
    target_snr_db,
    output_path
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        noisy_speech=noisy_speech,
        clean_speech=clean_speech,
        speech_filename=str(speech_file.name),
        noise_filename=str(noise_file.name),
        target_snr_db=target_snr_db
    )


def create_balanced_snr_list(
    snr_values,
    num_samples,
    rng
):

    if num_samples % len(snr_values) != 0:
        raise ValueError(
            "num_samples must be divisible by "
            "the number of SNR values."
        )

    samples_per_snr = (
        num_samples // len(snr_values)
    )

    balanced_snr_list = []

    for snr in snr_values:

        balanced_snr_list.extend(
            [snr] * samples_per_snr
        )

    rng.shuffle(
        balanced_snr_list
    )

    return balanced_snr_list

def generate_dataset(
    speech_files,
    noise_files,
    snr_values,
    num_samples,
    output_directory,
    rng
):

    balanced_snr_list = (
        create_balanced_snr_list(
            snr_values,
            num_samples,
            rng
        )
    )

    for index in range(num_samples):

        target_snr_db = balanced_snr_list[index]

        max_attempts = 10

        for attempt in range(max_attempts):

            speech_file = select_speech_file(
                speech_files,
                rng
            )

            noise_file = select_noise_file(
                noise_files,
                rng
            )

            try:

                noisy_speech, clean_speech, _ = (
                    generate_sample(
                        speech_file,
                        noise_file,
                        target_snr_db,
                        rng
                    )
                )

                break

            except ValueError as error:

                print(
                    f"Skipping invalid sample attempt "
                    f"{attempt + 1}/{max_attempts}: "
                    f"{error}"
                )

        else:

            raise RuntimeError(
                "Could not generate a valid sample "
                f"after {max_attempts} attempts."
            )

        output_path = (
            output_directory
            / f"sample_{index:06d}.npz"
        )

        save_sample(
            noisy_speech,
            clean_speech,
            speech_file,
            noise_file,
            target_snr_db,
            output_path
        )

        print(
            f"Generated {index + 1}/"
            f"{num_samples} | "
            f"SNR: {target_snr_db} dB"
        )


def main():

    PROJECT_ROOT = (
        Path(__file__).resolve().parent.parent
    )

    speech_path = (
        PROJECT_ROOT
        / "data"
        / "speech"
    )

    noise_path = (
        PROJECT_ROOT
        / "data"
        / "noise"
    )

    output_path = (
        PROJECT_ROOT
        / "processed_data"
    )

    speech_files = get_audio_files(
        speech_path
    )

    noise_files = get_audio_files(
        noise_path
    )

    print(
        f"Total speech files: {len(speech_files)}"
    )

    print(
        f"Total noise files: {len(noise_files)}"
    )

    train_files, validation_files, test_files = (
        split_dataset(
            speech_files,
            seed=42
        )
    )

    print("\nDataset Split")

    print(
        f"Training files: {len(train_files)}"
    )

    print(
        f"Validation files: {len(validation_files)}"
    )

    print(
        f"Test files: {len(test_files)}"
    )

    # Final experimental SNR levels
    snr_values = [-5, 0, 5, 10, 15]

    # Reproducible random generator
    rng = np.random.default_rng(42)

    print(
        "\nGenerating training dataset..."
    )

    generate_dataset(
        speech_files=train_files,
        noise_files=noise_files,
        snr_values=snr_values,
        num_samples=800,
        output_directory=(
            output_path / "train"
        ),
        rng=rng
    )

    print(
        "\nGenerating validation dataset..."
    )

    generate_dataset(
        speech_files=validation_files,
        noise_files=noise_files,
        snr_values=snr_values,
        num_samples=100,
        output_directory=(
            output_path / "validation"
        ),
        rng=rng
    )

    print(
        "\nGenerating test dataset..."
    )

    generate_dataset(
        speech_files=test_files,
        noise_files=noise_files,
        snr_values=snr_values,
        num_samples=100,
        output_directory=(
            output_path / "test"
        ),
        rng=rng
    )

    print(
        "\nFinal dataset generation completed."
    )


if __name__ == "__main__":
    main()
