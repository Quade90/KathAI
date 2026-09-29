import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.signal import (
    stft,
    find_peaks,
    iirnotch,
    sosfiltfilt,
    tf2sos
)

from pathlib import Path

from analyze_audio import analyze_audio


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

reference_path = BASE_DIR / "reference" / "reference.wav"
degraded_path = BASE_DIR / "degraded" / "hum_50hz.wav"

output_dir = BASE_DIR / "restored" / "hum"
output_dir.mkdir(parents=True, exist_ok=True)

restored_path = output_dir / "restored_hum.wav"
measurement_path = output_dir / "hum_restoration_measurements.txt"

plots_dir = BASE_DIR / "plots" / "restored_hum"


# ============================================================
# Load audio
# ============================================================

reference, sr_ref = sf.read(reference_path)
degraded, sr_deg = sf.read(degraded_path)

if sr_ref != sr_deg:
    raise ValueError(
        "Reference and degraded audio have different sample rates."
    )

sr = sr_ref


# ============================================================
# Convert stereo -> mono
# ============================================================

if reference.ndim > 1:
    reference = np.mean(reference, axis=1)

if degraded.ndim > 1:
    degraded = np.mean(degraded, axis=1)


# ============================================================
# Match lengths
# ============================================================

N = min(len(reference), len(degraded))

reference = reference[:N]
degraded = degraded[:N]


# ============================================================
# HUM DETECTION
# ============================================================
#
# We do NOT assume that the hum is 50 Hz.
#
# The detector searches from 20-300 Hz and looks for:
#
#   1. A narrow spectral peak
#   2. Strong contrast with neighboring frequencies
#   3. Persistence across time
#
# This distinguishes a continuous hum from ordinary speech
# frequency content.
# ============================================================


# ------------------------------------------------------------
# High-resolution FFT
# ------------------------------------------------------------

n_fft = 262144

spectrum = np.abs(
    np.fft.rfft(
        degraded,
        n=n_fft
    )
)

frequencies = np.fft.rfftfreq(
    n_fft,
    1 / sr
)


# ------------------------------------------------------------
# Search low-frequency region
# ------------------------------------------------------------

search_mask = (
    (frequencies >= 20) &
    (frequencies <= 300)
)

search_freqs = frequencies[search_mask]
search_spectrum = spectrum[search_mask]


# ------------------------------------------------------------
# Smooth spectrum slightly
# ------------------------------------------------------------

# This removes tiny FFT-bin fluctuations while preserving
# narrow peaks.

kernel_size = 5

kernel = np.ones(kernel_size) / kernel_size

smoothed_spectrum = np.convolve(
    search_spectrum,
    kernel,
    mode="same"
)


# ------------------------------------------------------------
# Find candidate spectral peaks
# ------------------------------------------------------------

peaks, properties = find_peaks(
    smoothed_spectrum,
    prominence=np.max(smoothed_spectrum) * 0.001,
    distance=10
)

if len(peaks) == 0:
    raise RuntimeError(
        "No low-frequency spectral peaks were detected."
    )


# ============================================================
# STFT FOR TEMPORAL PERSISTENCE
# ============================================================

nperseg = 16384
noverlap = 8192

stft_freqs, stft_times, Zxx = stft(
    degraded,
    fs=sr,
    nperseg=nperseg,
    noverlap=noverlap
)

stft_magnitude = np.abs(Zxx)


# ============================================================
# Score candidate frequencies
# ============================================================

candidate_data = []


for peak in peaks:

    candidate_frequency = search_freqs[peak]


    # --------------------------------------------------------
    # Spectral prominence
    # --------------------------------------------------------

    left = max(0, peak - 5)
    right = min(
        len(smoothed_spectrum),
        peak + 6
    )

    neighboring_values = np.concatenate(
        [
            smoothed_spectrum[left:peak],
            smoothed_spectrum[peak + 1:right]
        ]
    )

    if len(neighboring_values) == 0:
        continue

    neighbor_median = np.median(
        neighboring_values
    )

    peak_value = smoothed_spectrum[peak]

    contrast = (
        peak_value /
        (neighbor_median + 1e-12)
    )


    # --------------------------------------------------------
    # Find closest STFT frequency
    # --------------------------------------------------------

    stft_index = np.argmin(
        np.abs(
            stft_freqs -
            candidate_frequency
        )
    )

    candidate_track = (
        stft_magnitude[stft_index]
    )


    # --------------------------------------------------------
    # Temporal persistence
    # --------------------------------------------------------

    # A continuous hum should appear in most frames.
    #
    # Use a threshold relative to the candidate's own
    # temporal magnitude.

    threshold = np.percentile(
        candidate_track,
        40
    )

    persistence = np.mean(
        candidate_track > threshold
    )


    # --------------------------------------------------------
    # Temporal stability
    # --------------------------------------------------------

    # A real hum should not fluctuate wildly in frequency
    # content like voiced speech does.

    median_magnitude = np.median(
        candidate_track
    )

    mean_magnitude = np.mean(
        candidate_track
    )

    if mean_magnitude > 0:
        stability = (
            median_magnitude /
            mean_magnitude
        )
    else:
        stability = 0


    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    score = (
        np.log1p(contrast)
        *
        persistence
        *
        stability
    )


    candidate_data.append(
        (
            candidate_frequency,
            contrast,
            persistence,
            stability,
            score
        )
    )


if len(candidate_data) == 0:
    raise RuntimeError(
        "No valid hum candidates were found."
    )


# ============================================================
# Rank candidates
# ============================================================

candidate_data.sort(
    key=lambda x: x[4],
    reverse=True
)


# Best candidate
hum_frequency = candidate_data[0][0]


# ============================================================
# Print candidates
# ============================================================

print()
print("=" * 70)
print("HUM DETECTION")
print("=" * 70)

print()
print(
    f"{'Frequency':>12}"
    f"{'Contrast':>15}"
    f"{'Persistence':>15}"
    f"{'Stability':>15}"
    f"{'Score':>15}"
)

print("-" * 70)

for candidate in candidate_data[:10]:

    frequency = candidate[0]
    contrast = candidate[1]
    persistence = candidate[2]
    stability = candidate[3]
    score = candidate[4]

    print(
        f"{frequency:12.3f}"
        f"{contrast:15.3f}"
        f"{persistence:15.3f}"
        f"{stability:15.3f}"
        f"{score:15.3f}"
    )

print("-" * 70)

print(
    f"Detected hum frequency: "
    f"{hum_frequency:.3f} Hz"
)


# ============================================================
# Design notch filter
# ============================================================

Q = 30

b, a = iirnotch(
    w0=hum_frequency,
    Q=Q,
    fs=sr
)

sos = tf2sos(
    b,
    a
)


# ============================================================
# Apply notch filter
# ============================================================

restored = sosfiltfilt(
    sos,
    degraded
)


# ============================================================
# Save restored audio
# ============================================================

sf.write(
    restored_path,
    restored,
    sr
)


# ============================================================
# Quantitative comparison
# ============================================================

rmse_before = np.sqrt(
    np.mean(
        (reference - degraded) ** 2
    )
)

rmse_after = np.sqrt(
    np.mean(
        (reference - restored) ** 2
    )
)


correlation_before = np.corrcoef(
    reference,
    degraded
)[0, 1]

correlation_after = np.corrcoef(
    reference,
    restored
)[0, 1]


# ============================================================
# Frequency amplitude measurement
# ============================================================

def amplitude_at_frequency(
    signal,
    frequency,
    sr,
    n_fft=262144
):

    spectrum = np.abs(
        np.fft.rfft(
            signal,
            n=n_fft
        )
    )

    frequencies = np.fft.rfftfreq(
        n_fft,
        1 / sr
    )

    index = np.argmin(
        np.abs(
            frequencies -
            frequency
        )
    )

    return spectrum[index]


hum_before = amplitude_at_frequency(
    degraded,
    hum_frequency,
    sr
)

hum_after = amplitude_at_frequency(
    restored,
    hum_frequency,
    sr
)


# ============================================================
# Hum attenuation
# ============================================================

if hum_before > 0 and hum_after > 0:

    hum_attenuation_db = (
        20 *
        np.log10(
            hum_after /
            hum_before
        )
    )

else:

    hum_attenuation_db = float("-inf")


# ============================================================
# Save measurements
# ============================================================

with open(
    measurement_path,
    "w"
) as f:

    f.write(
        "HUM RESTORATION MEASUREMENTS\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        f"Reference file: "
        f"{reference_path.name}\n"
    )

    f.write(
        f"Degraded file: "
        f"{degraded_path.name}\n"
    )

    f.write(
        f"Restored file: "
        f"{restored_path.name}\n\n"
    )

    f.write(
        f"Sample rate (Hz): "
        f"{sr}\n"
    )

    f.write(
        f"Detected hum frequency (Hz): "
        f"{hum_frequency:.6f}\n"
    )

    f.write(
        f"Notch filter Q: "
        f"{Q}\n\n"
    )

    f.write(
        "REFERENCE MATCH\n"
    )

    f.write(
        "----------------\n"
    )

    f.write(
        f"RMSE before restoration: "
        f"{rmse_before:.6f}\n"
    )

    f.write(
        f"RMSE after restoration: "
        f"{rmse_after:.6f}\n"
    )

    f.write(
        f"Correlation before restoration: "
        f"{correlation_before:.6f}\n"
    )

    f.write(
        f"Correlation after restoration: "
        f"{correlation_after:.6f}\n\n"
    )

    f.write(
        "HUM REMOVAL\n"
    )

    f.write(
        "------------\n"
    )

    f.write(
        f"Hum amplitude before: "
        f"{hum_before:.6f}\n"
    )

    f.write(
        f"Hum amplitude after: "
        f"{hum_after:.6f}\n"
    )

    f.write(
        f"Hum attenuation (dB): "
        f"{hum_attenuation_db:.6f}\n"
    )


# ============================================================
# Waveform comparison
# ============================================================

time = np.arange(N) / sr

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time,
    reference,
    label="Reference",
    alpha=0.8
)

plt.plot(
    time,
    degraded,
    label="Hummed",
    alpha=0.6
)

plt.plot(
    time,
    restored,
    label="Restored",
    alpha=0.8
)

plt.title(
    "Hum Restoration: Waveform Comparison"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Amplitude"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    output_dir /
    "waveform_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# Spectrum comparison
# ============================================================

n_fft_plot = 262144

reference_spectrum = np.abs(
    np.fft.rfft(
        reference,
        n=n_fft_plot
    )
)

degraded_spectrum = np.abs(
    np.fft.rfft(
        degraded,
        n=n_fft_plot
    )
)

restored_spectrum = np.abs(
    np.fft.rfft(
        restored,
        n=n_fft_plot
    )
)

freqs = np.fft.rfftfreq(
    n_fft_plot,
    1 / sr
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    freqs,
    reference_spectrum,
    label="Reference",
    alpha=0.8
)

plt.plot(
    freqs,
    degraded_spectrum,
    label="Hummed",
    alpha=0.7
)

plt.plot(
    freqs,
    restored_spectrum,
    label="Restored",
    alpha=0.8
)

plt.xlim(
    0,
    500
)

plt.axvline(
    hum_frequency,
    linestyle="--",
    alpha=0.7,
    label=(
        f"Detected hum: "
        f"{hum_frequency:.2f} Hz"
    )
)

plt.title(
    "Hum Restoration: Low-Frequency Spectrum"
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Magnitude"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    output_dir /
    "spectrum_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# Standardized analysis
# ============================================================

analyze_audio(
    restored_path,
    plots_dir
)


# ============================================================
# Final output
# ============================================================

print()
print("=" * 70)
print("HUM RESTORATION RESULTS")
print("=" * 70)

print()

print(
    f"Detected hum frequency : "
    f"{hum_frequency:.6f} Hz"
)

print(
    f"Notch filter Q         : "
    f"{Q}"
)

print()

print("Reference comparison:")

print(
    f"RMSE before            : "
    f"{rmse_before:.6f}"
)

print(
    f"RMSE after             : "
    f"{rmse_after:.6f}"
)

print(
    f"Correlation before     : "
    f"{correlation_before:.6f}"
)

print(
    f"Correlation after      : "
    f"{correlation_after:.6f}"
)

print()

print("Hum removal:")

print(
    f"Hum amplitude before   : "
    f"{hum_before:.6f}"
)

print(
    f"Hum amplitude after    : "
    f"{hum_after:.6f}"
)

print(
    f"Hum attenuation        : "
    f"{hum_attenuation_db:.6f} dB"
)

print()

print("Files:")

print(
    f"Restored audio         : "
    f"{restored_path}"
)

print(
    f"Measurements           : "
    f"{measurement_path}"
)

print(
    f"Comparison plots       : "
    f"{output_dir}"
)

print(
    f"Standard plots         : "
    f"{plots_dir}"
)

print()

print("=" * 70)