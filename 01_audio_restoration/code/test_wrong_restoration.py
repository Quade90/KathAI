import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

reference_path = BASE_DIR / "reference" / "reference.wav"
degraded_path = BASE_DIR / "degraded" / "clipped_03.wav"

output_dir = BASE_DIR / "restored" / "wrong_dereverb_clipping"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

restored_path = (
    output_dir / "restored_clipped_with_dereverb.wav"
)

measurement_path = (
    output_dir /
    "wrong_restoration_measurements.txt"
)


# ============================================================
# Parameters
# ============================================================

REGULARIZATION = 0.001


# ============================================================
# Load audio
# ============================================================

def load_mono(path):

    y, sr = sf.read(path)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    return y.astype(np.float64), sr


# ============================================================
# Create assumed room impulse response
# ============================================================

def create_room_impulse_response(sr):

    """
    Same room impulse response used for the normal
    dereverberation experiment.

    The restoration method is deliberately being applied
    to a clipped signal, for which this model is incorrect.
    """

    ir_duration = 0.8
    ir_length = int(
        ir_duration * sr
    )

    ir = np.zeros(ir_length)

    # Direct sound
    ir[0] = 1.0

    # Early reflections
    reflections = [
        (0.020, 0.5),
        (0.045, 0.3),
        (0.080, 0.2),
        (0.120, 0.1),
    ]

    for delay, amplitude in reflections:

        index = int(
            delay * sr
        )

        if index < ir_length:
            ir[index] = amplitude

    # Decaying reverb tail
    tail_start = int(
        0.15 * sr
    )

    rng = np.random.default_rng(42)

    tail_length = (
        ir_length - tail_start
    )

    t = (
        np.arange(tail_length)
        / sr
    )

    decay = np.exp(
        -5.0 * t
    )

    noise = rng.normal(
        0,
        1,
        tail_length
    )

    ir[tail_start:] += (
        0.05
        * noise
        * decay
    )

    return ir


# ============================================================
# Regularized inverse filter
# ============================================================

def regularized_inverse_filter(
    degraded,
    ir,
    regularization
):

    """
    Regularized inverse filter:

                H*(f)
        G(f) = -----------
                |H(f)|² + λ

    This is the dereverberation method used in Q5.

    It is deliberately applied to clipping here,
    even though clipping is not a convolutional degradation.
    """

    n = len(degraded)

    convolution_length = (
        len(degraded)
        + len(ir)
        - 1
    )

    fft_length = 2 ** int(
        np.ceil(
            np.log2(convolution_length)
        )
    )

    Y = np.fft.rfft(
        degraded,
        n=fft_length
    )

    H = np.fft.rfft(
        ir,
        n=fft_length
    )

    denominator = (
        np.abs(H) ** 2
        + regularization
    )

    G = (
        np.conj(H)
        / denominator
    )

    restored_spectrum = (
        Y * G
    )

    restored = np.fft.irfft(
        restored_spectrum,
        n=fft_length
    )

    restored = restored[:n]

    return restored


# ============================================================
# Metrics
# ============================================================

def rmse(a, b):

    return np.sqrt(
        np.mean(
            (a - b) ** 2
        )
    )


def correlation(a, b):

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return (
        np.dot(a, b)
        / denominator
    )


def rms(x):

    return np.sqrt(
        np.mean(x ** 2)
    )


# ============================================================
# Load reference and clipped signal
# ============================================================

print("Loading audio...")

reference, sr_reference = load_mono(
    reference_path
)

clipped, sr_clipped = load_mono(
    degraded_path
)

if sr_reference != sr_clipped:

    raise RuntimeError(
        "Reference and clipped audio have "
        "different sample rates."
    )

sr = sr_reference

length = min(
    len(reference),
    len(clipped)
)

reference = reference[:length]
clipped = clipped[:length]

print(
    f"Sample rate : {sr} Hz"
)

print(
    f"Duration    : "
    f"{length / sr:.3f} s"
)


# ============================================================
# Measure clipping before restoration
# ============================================================

clipping_fraction_before = np.mean(
    np.abs(clipped) >= 0.299
)

print()
print(
    "Clipping signal:"
)

print(
    f"Peak : "
    f"{np.max(np.abs(clipped)):.6f}"
)

print(
    f"RMS  : "
    f"{rms(clipped):.6f}"
)

print(
    f"Samples near clipping threshold: "
    f"{clipping_fraction_before * 100:.4f}%"
)


# ============================================================
# Create room impulse response
# ============================================================

print()
print(
    "Creating assumed room impulse response..."
)

ir = create_room_impulse_response(
    sr
)


# ============================================================
# Apply WRONG restoration method
# ============================================================

print()
print(
    "Applying dereverberation to clipped audio..."
)

print(
    f"Regularization = "
    f"{REGULARIZATION}"
)

restored = regularized_inverse_filter(
    clipped,
    ir,
    REGULARIZATION
)


# ============================================================
# Remove DC
# ============================================================

restored -= np.mean(
    restored
)


# ============================================================
# Check validity
# ============================================================

if not np.all(
    np.isfinite(restored)
):

    raise RuntimeError(
        "Restoration produced NaN or infinite values."
    )


# ============================================================
# Safe normalization
# ============================================================

peak = np.max(
    np.abs(restored)
)

print()
print(
    "Wrongly restored signal before normalization:"
)

print(
    f"Peak : {peak:.6f}"
)

print(
    f"RMS  : {rms(restored):.6f}"
)

if peak == 0:

    raise RuntimeError(
        "Restoration produced an empty signal."
    )

if peak > 0.95:

    restored *= (
        0.9 / peak
    )

restored = np.clip(
    restored,
    -0.999,
    0.999
)


# ============================================================
# Save
# ============================================================

sf.write(
    restored_path,
    restored,
    sr,
    subtype="PCM_16"
)


# ============================================================
# Read saved WAV back
# ============================================================

saved, saved_sr = load_mono(
    restored_path
)

print()
print(
    "Saved WAV validation:"
)

print(
    f"Sample rate : "
    f"{saved_sr} Hz"
)

print(
    f"Min         : "
    f"{np.min(saved):.6f}"
)

print(
    f"Max         : "
    f"{np.max(saved):.6f}"
)

print(
    f"Peak        : "
    f"{np.max(np.abs(saved)):.6f}"
)

print(
    f"RMS         : "
    f"{rms(saved):.6f}"
)

print(
    f"Mean        : "
    f"{np.mean(saved):.8f}"
)


# ============================================================
# Compare with reference
# ============================================================

reference_aligned = reference[
    :len(saved)
]

clipped_aligned = clipped[
    :len(saved)
]

restored_aligned = saved[
    :len(reference_aligned)
]

reference_aligned = reference_aligned[
    :len(restored_aligned)
]

clipped_aligned = clipped_aligned[
    :len(restored_aligned)
]


# ============================================================
# Calculate metrics
# ============================================================

rmse_before = rmse(
    reference_aligned,
    clipped_aligned
)

rmse_after = rmse(
    reference_aligned,
    restored_aligned
)

correlation_before = correlation(
    reference_aligned,
    clipped_aligned
)

correlation_after = correlation(
    reference_aligned,
    restored_aligned
)

rms_reference = rms(
    reference_aligned
)

rms_clipped = rms(
    clipped_aligned
)

rms_restored = rms(
    restored_aligned
)


# ============================================================
# Measure remaining clipping
# ============================================================

clipping_fraction_after = np.mean(
    np.abs(restored_aligned) >= 0.299
)


# ============================================================
# Save measurements
# ============================================================

with open(
    measurement_path,
    "w"
) as f:

    f.write(
        "WRONG RESTORATION: DEREVERBERATION ON CLIPPING\n"
    )

    f.write(
        "================================================\n\n"
    )

    f.write(
        "Purpose:\n"
    )

    f.write(
        "Apply a dereverberation method to a clipped signal "
        "for Q6.\n\n"
    )

    f.write(
        f"Sample rate: {sr} Hz\n"
    )

    f.write(
        f"Regularization: "
        f"{REGULARIZATION}\n\n"
    )

    f.write(
        "Reference vs clipped\n"
    )

    f.write(
        "--------------------\n"
    )

    f.write(
        f"RMSE: "
        f"{rmse_before:.6f}\n"
    )

    f.write(
        f"Correlation: "
        f"{correlation_before:.6f}\n\n"
    )

    f.write(
        "Reference vs wrongly restored\n"
    )

    f.write(
        "-----------------------------\n"
    )

    f.write(
        f"RMSE: "
        f"{rmse_after:.6f}\n"
    )

    f.write(
        f"Correlation: "
        f"{correlation_after:.6f}\n\n"
    )

    f.write(
        "RMS comparison\n"
    )

    f.write(
        "--------------\n"
    )

    f.write(
        f"Reference RMS: "
        f"{rms_reference:.6f}\n"
    )

    f.write(
        f"Clipped RMS: "
        f"{rms_clipped:.6f}\n"
    )

    f.write(
        f"Restored RMS: "
        f"{rms_restored:.6f}\n\n"
    )

    f.write(
        "Clipping comparison\n"
    )

    f.write(
        "-------------------\n"
    )

    f.write(
        f"Before restoration: "
        f"{clipping_fraction_before * 100:.6f}%\n"
    )

    f.write(
        f"After restoration: "
        f"{clipping_fraction_after * 100:.6f}%\n"
    )


# ============================================================
# Waveform comparison
# ============================================================

time = (
    np.arange(
        len(reference_aligned)
    )
    / sr
)

plt.figure(
    figsize=(14, 8)
)

plt.subplot(3, 1, 1)

plt.plot(
    time,
    reference_aligned
)

plt.title(
    "Reference"
)

plt.ylabel(
    "Amplitude"
)

plt.grid(
    alpha=0.3
)


plt.subplot(3, 1, 2)

plt.plot(
    time,
    clipped_aligned
)

plt.title(
    "Clipped"
)

plt.ylabel(
    "Amplitude"
)

plt.grid(
    alpha=0.3
)


plt.subplot(3, 1, 3)

plt.plot(
    time,
    restored_aligned
)

plt.title(
    "Dereverberation Applied to Clipped Signal"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Amplitude"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    output_dir / "waveform_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# Spectrum comparison
# ============================================================

N_FFT = 8192

reference_spectrum = np.abs(
    np.fft.rfft(
        reference_aligned,
        n=N_FFT
    )
)

clipped_spectrum = np.abs(
    np.fft.rfft(
        clipped_aligned,
        n=N_FFT
    )
)

restored_spectrum = np.abs(
    np.fft.rfft(
        restored_aligned,
        n=N_FFT
    )
)

frequencies = np.fft.rfftfreq(
    N_FFT,
    1 / sr
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    frequencies,
    reference_spectrum,
    label="Reference"
)

plt.plot(
    frequencies,
    clipped_spectrum,
    label="Clipped"
)

plt.plot(
    frequencies,
    restored_spectrum,
    label="Wrongly restored"
)

plt.xlim(
    0,
    min(
        10000,
        sr / 2
    )
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Magnitude"
)

plt.title(
    "Wrong Restoration Spectrum Comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    output_dir / "spectrum_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# Final output
# ============================================================

print()
print(
    "=" * 60
)

print(
    "WRONG RESTORATION TEST COMPLETE"
)

print(
    "=" * 60
)

print()

print(
    "Reference comparison:"
)

print(
    f"RMSE before       : "
    f"{rmse_before:.6f}"
)

print(
    f"RMSE after        : "
    f"{rmse_after:.6f}"
)

print(
    f"Correlation before: "
    f"{correlation_before:.6f}"
)

print(
    f"Correlation after : "
    f"{correlation_after:.6f}"
)

print()

print(
    "RMS:"
)

print(
    f"Reference : "
    f"{rms_reference:.6f}"
)

print(
    f"Clipped   : "
    f"{rms_clipped:.6f}"
)

print(
    f"Restored  : "
    f"{rms_restored:.6f}"
)

print()

print(
    f"Restored audio : "
    f"{restored_path}"
)

print(
    f"Measurements   : "
    f"{measurement_path}"
)

print(
    f"Plots          : "
    f"{output_dir}"
)