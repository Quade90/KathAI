
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

from pathlib import Path


def analyze_audio(input_path, output_dir):

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # Load audio
    # ==================================================

    y, sr = sf.read(input_path)

    # Convert stereo to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    duration = len(y) / sr

    # ==================================================
    # Basic measurements
    # ==================================================

    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y ** 2))

    # ==================================================
    # Spectral measurements
    # ==================================================

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    centroid_mean = np.mean(spectral_centroid)
    bandwidth_mean = np.mean(spectral_bandwidth)
    rolloff_mean = np.mean(spectral_rolloff)

    # ==================================================
    # Save measurements
    # ==================================================

    measurements = {
        "Duration (s)": duration,
        "Sample rate (Hz)": sr,
        "Samples": len(y),
        "Peak amplitude": peak,
        "RMS amplitude": rms,
        "Spectral centroid (Hz)": centroid_mean,
        "Spectral bandwidth (Hz)": bandwidth_mean,
        "Spectral rolloff (Hz)": rolloff_mean,
    }

    measurements_path = output_dir / "measurement.txt"

    with open(measurements_path, "w") as f:

        f.write("AUDIO MEASUREMENTS\n")
        f.write("==================\n\n")

        f.write(f"Input file: {input_path.name}\n\n")

        for name, value in measurements.items():

            if isinstance(value, (float, np.floating)):
                f.write(f"{name}: {value:.6f}\n")
            else:
                f.write(f"{name}: {value}\n")

    # ==================================================
    # 1. Waveform
    # ==================================================

    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(
        y,
        sr=sr
    )

    plt.title("Speech Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    plt.savefig(
        output_dir / "waveform.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # 2. Frequency spectrum (FULL RECORDING)
    # ==================================================

    # Use the complete signal instead of truncating
    # it to the first 4096 samples.

    n_fft = len(y)

    spectrum = np.abs(
        np.fft.rfft(y)
    )

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1 / sr
    )

    plt.figure(figsize=(12, 4))

    plt.plot(
        frequencies,
        spectrum,
        linewidth=0.7
    )

    plt.xlim(
        0,
        min(10000, sr / 2)
    )

    plt.title("Speech Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "spectrum.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # 3. Spectrogram
    # ==================================================

    D = librosa.stft(
        y,
        n_fft=2048,
        hop_length=512
    )

    D_db = librosa.amplitude_to_db(
        np.abs(D),
        ref=np.max
    )

    plt.figure(figsize=(12, 6))

    librosa.display.specshow(
        D_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.ylim(
        0,
        min(10000, sr / 2)
    )

    plt.title("Speech Spectrogram")

    plt.tight_layout()

    plt.savefig(
        output_dir / "spectrogram.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # Print measurements
    # ==================================================

    print(f"\nAnalysis: {input_path.name}")
    print("-" * 40)

    for name, value in measurements.items():

        if isinstance(value, (float, np.floating)):
            print(f"{name:<30}: {value:.6f}")
        else:
            print(f"{name:<30}: {value}")

    print(f"\nSaved to: {output_dir}")
