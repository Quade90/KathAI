import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from pathlib import Path

from analyze_audio import analyze_audio


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

reference_path = BASE_DIR / "reference" / "reference.wav"
degraded_path = BASE_DIR / "degraded" / "reverb.wav"

output_dir = BASE_DIR / "restored" / "dereverb"
plots_dir = BASE_DIR / "plots" / "restored_dereverb"

output_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

restored_path = output_dir / "restored_dereverb.wav"
measurement_path = (
    output_dir / "dereverb_restoration_measurements.txt"
)


# ============================================================
# Regularization
# ============================================================

# Larger lambda = safer, but less aggressive inversion.
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
# Build the assumed room impulse response
# ============================================================

def create_room_impulse_response(sr):
    """
    Reconstruct the synthetic room impulse response used to
    create the reverberant recording.

    This is treated as an estimated/known room response.
    The clean reference signal is NOT used here.
    """

    ir_duration = 0.8
    ir_length = int(ir_duration * sr)

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

        index = int(delay * sr)

        if index < ir_length:
            ir[index] = amplitude

    # Decaying diffuse tail
    tail_start = int(0.15 * sr)

    rng = np.random.default_rng(42)

    tail_length = ir_length - tail_start

    t = np.arange(tail_length) / sr

    decay = np.exp(-5.0 * t)

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
# Regularized inverse filtering
# ============================================================

def regularized_inverse_filter(
    degraded,
    ir,
    regularization
):

    n = len(degraded)

    # Linear convolution length
    convolution_length = (
        len(degraded) + len(ir) - 1
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

    # --------------------------------------------------------
    # Wiener/Tikhonov-style regularized inverse
    #
    # G(f) = H*(f) / (|H(f)|² + λ)
    # --------------------------------------------------------

    H_conjugate = np.conj(H)

    denominator = (
        np.abs(H) ** 2
        + regularization
    )

    G = H_conjugate / denominator

    # Apply inverse filter
    restored_spectrum = Y * G

    restored = np.fft.irfft(
        restored_spectrum,
        n=fft_length
    )

    # Keep original recording length
    restored = restored[:n]

    return restored


# ============================================================
# Load reference and degraded audio
# ============================================================

print("Loading audio...")

reference, sr_reference = load_mono(
    reference_path
)

degraded, sr_degraded = load_mono(
    degraded_path
)

if sr_reference != sr_degraded:

    raise RuntimeError(
        "Reference and degraded audio have "
        "different sample rates."
    )

sr = sr_reference

# Align lengths
length = min(
    len(reference),
    len(degraded)
)

reference = reference[:length]
degraded = degraded[:length]

print(
    f"Sample rate : {sr} Hz"
)

print(
    f"Duration    : "
    f"{length / sr:.3f} s"
)


# ============================================================
# Create room impulse response
# ============================================================

print()
print(
    "Using known/estimated room impulse response..."
)

ir = create_room_impulse_response(sr)

print(
    f"IR duration : "
    f"{len(ir) / sr:.3f} s"
)


# ============================================================
# Plot impulse response
# ============================================================

ir_time = np.arange(len(ir)) / sr

plt.figure(figsize=(12, 4))

plt.plot(
    ir_time,
    ir
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(
    "Room Impulse Response Used for Restoration"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_dir / "room_impulse_response.png",
    dpi=300
)

plt.close()


# ============================================================
# Apply regularized inverse filter
# ============================================================

print()
print(
    "Applying regularized inverse filter..."
)

restored = regularized_inverse_filter(
    degraded,
    ir,
    REGULARIZATION
)


# ============================================================
# Remove DC offset
# ============================================================

restored -= np.mean(restored)


# ============================================================
# Check numerical validity
# ============================================================

if not np.all(
    np.isfinite(restored)
):

    raise RuntimeError(
        "Restoration produced NaN or infinite values."
    )


# ============================================================
# Normalize safely
# ============================================================

peak = np.max(
    np.abs(restored)
)

print()
print(
    "Restored signal before normalization:"
)

print(
    f"Peak : {peak:.6f}"
)

print(
    f"RMS  : "
    f"{np.sqrt(np.mean(restored ** 2)):.6f}"
)

if peak == 0:

    raise RuntimeError(
        "Restoration produced an empty signal."
    )

# Only normalize if necessary.
if peak > 0.95:

    restored *= 0.9 / peak

# Final safety limit
restored = np.clip(
    restored,
    -0.999,
    0.999
)


# ============================================================
# Save restored WAV
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
    f"Sample rate : {saved_sr} Hz"
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
    f"RMS         : "
    f"{np.sqrt(np.mean(saved ** 2)):.6f}"
)

print(
    f"Mean        : "
    f"{np.mean(saved):.8f}"
)

negative_clipping = np.mean(
    saved <= -0.998
)

positive_clipping = np.mean(
    saved >= 0.998
)

print(
    f"Samples at -1 : "
    f"{negative_clipping * 100:.4f}%"
)

print(
    f"Samples at +1 : "
    f"{positive_clipping * 100:.4f}%"
)

if negative_clipping > 0.01:

    raise RuntimeError(
        "Restored signal has excessive negative clipping."
    )

if positive_clipping > 0.01:

    raise RuntimeError(
        "Restored signal has excessive positive clipping."
    )


# ============================================================
# Compare reference / degraded / restored
# ============================================================

reference_aligned = reference[:len(saved)]
degraded_aligned = degraded[:len(saved)]
restored_aligned = saved[:len(reference_aligned)]

reference_aligned = reference_aligned[
    :len(restored_aligned)
]

degraded_aligned = degraded_aligned[
    :len(restored_aligned)
]


# ============================================================
# Metrics
# ============================================================

rmse_before = np.sqrt(
    np.mean(
        (
            reference_aligned
            - degraded_aligned
        ) ** 2
    )
)

rmse_after = np.sqrt(
    np.mean(
        (
            reference_aligned
            - restored_aligned
        ) ** 2
    )
)


def calculate_correlation(a, b):

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


correlation_before = calculate_correlation(
    reference_aligned,
    degraded_aligned
)

correlation_after = calculate_correlation(
    reference_aligned,
    restored_aligned
)


def calculate_rms(x):

    return np.sqrt(
        np.mean(x ** 2)
    )


rms_reference = calculate_rms(
    reference_aligned
)

rms_degraded = calculate_rms(
    degraded_aligned
)

rms_restored = calculate_rms(
    restored_aligned
)


# ============================================================
# Measurements file
# ============================================================

with open(
    measurement_path,
    "w"
) as f:

    f.write(
        "DEREVERBERATION RESTORATION MEASUREMENTS\n"
    )

    f.write(
        "=========================================\n\n"
    )

    f.write(
        f"Sample rate: {sr} Hz\n"
    )

    f.write(
        f"Duration: "
        f"{len(restored_aligned) / sr:.6f} s\n"
    )

    f.write(
        f"Regularization: "
        f"{REGULARIZATION}\n\n"
    )

    f.write(
        "Reference vs degraded\n"
    )

    f.write(
        "---------------------\n"
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
        "Reference vs restored\n"
    )

    f.write(
        "---------------------\n"
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
        f"Degraded RMS: "
        f"{rms_degraded:.6f}\n"
    )

    f.write(
        f"Restored RMS: "
        f"{rms_restored:.6f}\n"
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

plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)

plt.plot(
    time,
    reference_aligned
)

plt.title("Reference")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)


plt.subplot(3, 1, 2)

plt.plot(
    time,
    degraded_aligned
)

plt.title("Reverberant")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)


plt.subplot(3, 1, 3)

plt.plot(
    time,
    restored_aligned
)

plt.title("Restored")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_dir / "waveform_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# Spectrum comparison
# ============================================================

N_FFT_PLOT = 8192

reference_spectrum = np.abs(
    np.fft.rfft(
        reference_aligned,
        n=N_FFT_PLOT
    )
)

degraded_spectrum = np.abs(
    np.fft.rfft(
        degraded_aligned,
        n=N_FFT_PLOT
    )
)

restored_spectrum = np.abs(
    np.fft.rfft(
        restored_aligned,
        n=N_FFT_PLOT
    )
)

frequencies = np.fft.rfftfreq(
    N_FFT_PLOT,
    1 / sr
)

plt.figure(figsize=(14, 6))

plt.plot(
    frequencies,
    reference_spectrum,
    label="Reference"
)

plt.plot(
    frequencies,
    degraded_spectrum,
    label="Reverberant"
)

plt.plot(
    frequencies,
    restored_spectrum,
    label="Restored"
)

plt.xlim(
    0,
    min(10000, sr / 2)
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.title(
    "Frequency Spectrum Comparison"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_dir / "spectrum_comparison.png",
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
# Final results
# ============================================================

print()
print("=" * 60)
print(
    "DEREVERBERATION COMPLETE"
)
print("=" * 60)

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

print("RMS:")

print(
    f"Reference : "
    f"{rms_reference:.6f}"
)

print(
    f"Degraded  : "
    f"{rms_degraded:.6f}"
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

print(
    f"Analysis       : "
    f"{plots_dir}"
)