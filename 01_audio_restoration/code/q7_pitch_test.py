import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "reference" / "reference.wav"
output_dir = BASE_DIR / "restored" / "q7_pitch"
output_dir.mkdir(parents=True, exist_ok=True)

noisy_path = output_dir / "noisy_rumble.wav"
processed_path = output_dir / "cleaned_highpass.wav"
measurement_path = output_dir / "q7_measurements.txt"

# --------------------------------------------------
# Load reference
# --------------------------------------------------

y, sr = sf.read(input_path)

if y.ndim > 1:
    y = np.mean(y, axis=1)

y = y.astype(np.float64)

# --------------------------------------------------
# Add low-frequency rumble
# --------------------------------------------------

duration = len(y) / sr
t = np.arange(len(y)) / sr

# Low-frequency components representing rumble
rumble = (
    0.10 * np.sin(2 * np.pi * 80 * t)
    + 0.07 * np.sin(2 * np.pi * 120 * t)
    + 0.04 * np.sin(2 * np.pi * 160 * t)
)

# Add a small amount of broadband noise
rng = np.random.default_rng(42)
noise = 0.015 * rng.normal(size=len(y))

y_noisy = y + rumble + noise
y_noisy = y_noisy / np.max(np.abs(y_noisy)) * 0.95

sf.write(noisy_path, y_noisy, sr)

# --------------------------------------------------
# Aggressive high-pass filtering
# --------------------------------------------------

# This removes the low-frequency rumble,
# but also removes useful speech information.
cutoff = 250
order = 6

sos = butter(
    order,
    cutoff,
    btype="highpass",
    fs=sr,
    output="sos"
)

y_clean = sosfiltfilt(sos, y_noisy)

# Safe normalization
if np.max(np.abs(y_clean)) > 0.95:
    y_clean = y_clean / np.max(np.abs(y_clean)) * 0.95

sf.write(processed_path, y_clean, sr)

# --------------------------------------------------
# Pitch tracking
# --------------------------------------------------

def track_pitch(signal, sr):
    f0, voiced_flag, voiced_prob = librosa.pyin(
        signal,
        fmin=70,
        fmax=400,
        sr=sr,
        frame_length=4096,
        hop_length=256
    )

    valid = ~np.isnan(f0)

    voiced_fraction = np.mean(valid)

    if np.any(valid):
        median_f0 = np.median(f0[valid])
        mean_confidence = np.mean(voiced_prob[valid])
    else:
        median_f0 = np.nan
        mean_confidence = 0.0

    return f0, voiced_fraction, median_f0, mean_confidence


f0_reference, ref_voiced, ref_median, ref_conf = track_pitch(y, sr)
f0_noisy, noisy_voiced, noisy_median, noisy_conf = track_pitch(y_noisy, sr)
f0_clean, clean_voiced, clean_median, clean_conf = track_pitch(y_clean, sr)

# --------------------------------------------------
# Basic measurements
# --------------------------------------------------

rms_reference = np.sqrt(np.mean(y ** 2))
rms_noisy = np.sqrt(np.mean(y_noisy ** 2))
rms_clean = np.sqrt(np.mean(y_clean ** 2))

# Spectral centroid
centroid_noisy = np.mean(
    librosa.feature.spectral_centroid(y=y_noisy, sr=sr)
)

centroid_clean = np.mean(
    librosa.feature.spectral_centroid(y=y_clean, sr=sr)
)

# --------------------------------------------------
# Save measurements
# --------------------------------------------------

with open(measurement_path, "w") as f:

    f.write("Q7: CLEANER AUDIO VS INFORMATION LOSS\n")
    f.write("=====================================\n\n")

    f.write(f"Sample rate: {sr} Hz\n")
    f.write(f"High-pass cutoff: {cutoff} Hz\n")
    f.write(f"Filter order: {order}\n\n")

    f.write("RMS\n")
    f.write("---\n")
    f.write(f"Reference: {rms_reference:.6f}\n")
    f.write(f"Noisy: {rms_noisy:.6f}\n")
    f.write(f"Processed: {rms_clean:.6f}\n\n")

    f.write("Pitch tracking\n")
    f.write("--------------\n")
    f.write(f"Reference voiced fraction: {ref_voiced:.6f}\n")
    f.write(f"Noisy voiced fraction: {noisy_voiced:.6f}\n")
    f.write(f"Processed voiced fraction: {clean_voiced:.6f}\n\n")

    f.write(f"Reference median F0: {ref_median:.3f} Hz\n")
    f.write(f"Noisy median F0: {noisy_median:.3f} Hz\n")
    f.write(f"Processed median F0: {clean_median:.3f} Hz\n\n")

    f.write(f"Reference pitch confidence: {ref_conf:.6f}\n")
    f.write(f"Noisy pitch confidence: {noisy_conf:.6f}\n")
    f.write(f"Processed pitch confidence: {clean_conf:.6f}\n\n")

    f.write("Spectral centroid\n")
    f.write("-----------------\n")
    f.write(f"Noisy: {centroid_noisy:.3f} Hz\n")
    f.write(f"Processed: {centroid_clean:.3f} Hz\n")

# --------------------------------------------------
# Plot pitch tracking
# --------------------------------------------------

times_ref = librosa.times_like(f0_reference, sr=sr, hop_length=256)
times_noisy = librosa.times_like(f0_noisy, sr=sr, hop_length=256)
times_clean = librosa.times_like(f0_clean, sr=sr, hop_length=256)

plt.figure(figsize=(12, 5))

plt.plot(
    times_ref,
    f0_reference,
    label="Reference"
)

plt.plot(
    times_noisy,
    f0_noisy,
    label="Noisy"
)

plt.plot(
    times_clean,
    f0_clean,
    label="Processed"
)

plt.xlabel("Time (s)")
plt.ylabel("Estimated F0 (Hz)")
plt.title("Pitch Tracking Comparison")
plt.ylim(50, 450)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "pitch_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Waveform comparison
# --------------------------------------------------

plt.figure(figsize=(12, 5))

time = np.arange(len(y)) / sr

plt.plot(time, y, label="Reference")
plt.plot(time, y_noisy, label="Noisy")
plt.plot(time, y_clean, label="Processed")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "waveform_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nQ7 RESULTS")
print("=" * 50)

print(f"High-pass cutoff       : {cutoff} Hz")

print("\nPitch tracking:")
print(f"Reference voiced fraction : {ref_voiced:.4f}")
print(f"Noisy voiced fraction     : {noisy_voiced:.4f}")
print(f"Processed voiced fraction : {clean_voiced:.4f}")

print(f"\nReference median F0 : {ref_median:.2f} Hz")
print(f"Noisy median F0     : {noisy_median:.2f} Hz")
print(f"Processed median F0 : {clean_median:.2f} Hz")

print(f"\nReference confidence : {ref_conf:.4f}")
print(f"Noisy confidence     : {noisy_conf:.4f}")
print(f"Processed confidence : {clean_conf:.4f}")

print("\nSaved:")
print(noisy_path)
print(processed_path)
print(measurement_path)
print(output_dir / "pitch_comparison.png")
print(output_dir / "waveform_comparison.png")