from pathlib import Path
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".flac"}

def get_audio_files(dataset_path):
    
    audio_files = [
        file
        for file in dataset_path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    return audio_files

def inspect_audio_file(audio_path):

    info = sf.info(audio_path)

    sample_rate = info.samplerate
    channels = info.channels
    frames = info.frames
    duration = frames / sample_rate

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration": duration
    }

def inspect_dataset(dataset_path, dataset_name):

    print("\n" + "=" * 50)
    print(f"Inspecting: {dataset_name}")
    print("=" * 50)

    if not dataset_path.exists():
        print("Dataset folder not found.")
        return

    audio_files = get_audio_files(dataset_path)

    if not audio_files:
        print("No audio files found.")
        return

    sample_rates = set()
    channel_counts = set()
    total_duration = 0
    unreadable_files = []

    for audio_file in audio_files:

        try:
            metadata = inspect_audio_file(audio_file)

            sample_rates.add(metadata["sample_rate"])
            channel_counts.add(metadata["channels"])
            total_duration += metadata["duration"]

        except Exception:
            unreadable_files.append(audio_file)

    print(f"Total audio files: {len(audio_files)}")
    print(f"Sample rates found: {sorted(sample_rates)}")
    print(f"Channel counts found: {sorted(channel_counts)}")
    print(f"Total duration: {total_duration / 60:.2f} minutes")
    print(f"Unreadable files: {len(unreadable_files)}")

def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    speech_path = PROJECT_ROOT / "data" / "speech"
    noise_path = PROJECT_ROOT / "data" / "noise"

    inspect_dataset(
        speech_path,
        "Speech Dataset"
    )

    inspect_dataset(
        noise_path,
        "Noise Dataset"
    )


if __name__ == "__main__":
    main()

