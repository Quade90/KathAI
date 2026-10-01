# 2.1 — The Damaged Recording

This section investigates how common recording degradations affect speech, what information survives each degradation, and when signal restoration is trustworthy.

The same short reference recording is used throughout the experiments.

## Overview

The pipeline covers:

1. baseline speech analysis
2. five controlled degradations
3. comparison of recoverable and irreversible information loss
4. two substantially different restoration methods
5. quantitative restoration evaluation
6. deliberately mismatched restoration
7. cleaner-sounding audio versus signal fidelity

## Code

```text
code/
├── analyze_audio.py
├── baseline.py
├── add_hum.py
├── bandpass.py
├── clipping.py
├── reverb.py
├── lossy_compress.py
├── restore_hum.py
├── restore_dereverb.py
├── test_wrong_restoration.py
└── q7_pitch_test.py
```

Generated audio, plots, measurements, and the personal reference recording are not committed to the repository.

## Input

Place the reference recording at:

```text
reference/reference.wav
```

The analysis used a recording of approximately 15 seconds, with speech activity mainly between roughly 0.7 s and 14.5 s.

## Baseline Analysis

`analyze_audio.py` is the shared analysis utility.

For each signal it calculates or produces:

- waveform
- full-recording magnitude spectrum
- STFT spectrogram
- duration
- sample rate
- peak amplitude
- RMS amplitude
- mean spectral centroid
- mean spectral bandwidth
- mean spectral rolloff

The spectrogram uses:

- `n_fft = 2048`
- `hop_length = 512`

Run:

```bash
python code/baseline.py
```

## Degradations

### 1. 50 Hz Hum

`add_hum.py`

A sinusoidal interference component is added to the reference signal:

- frequency: 50 Hz
- amplitude: 0.3

Run:

```bash
python code/add_hum.py
```

### 2. 500–4000 Hz Bandpass

`bandpass.py`

A sixth-order Butterworth bandpass filter keeps frequencies from 500 Hz to 4000 Hz while attenuating frequencies outside the passband.

Run:

```bash
python code/bandpass.py
```

### 3. Hard Clipping

`clipping.py`

The waveform is clipped at:

```text
±0.3
```

This is a nonlinear degradation: samples above the threshold lose their original amplitude information and additional harmonics are introduced.

Run:

```bash
python code/clipping.py
```

The generated file is `clipped_03.wav`.

### 4. Reverberation

`reverb.py`

Reverberation is produced by convolving the reference recording with a synthetic room impulse response containing:

- direct sound
- early reflections
- an exponentially decaying diffuse tail

The impulse response lasts 0.8 s.

Run:

```bash
python code/reverb.py
```

### 5. Low-Bitrate Lossy Compression

`lossy_compress.py`

The reference audio is encoded as a 32 kbps MP3 using FFmpeg and then decoded back to WAV for comparison.

Run:

```bash
python code/lossy_compress.py
```

> `lossy_compress.py` contains a local Windows path to FFmpeg. Change it if FFmpeg is installed elsewhere.

## Recoverable vs Irrecoverable Damage

Two useful contrasting cases are:

### Hum

The 50 Hz component is added to the original recording. Most speech information remains present, so a narrow notch can remove the interference while preserving most of the speech spectrum.

### Bandpass Filtering

The bandpass filter removes frequencies outside 500–4000 Hz. Those components are no longer present in the degraded recording, so exact reconstruction is not possible from the filtered signal alone.

This distinguishes **added interference** from **information that has actually been discarded**.

## Restoration 1 — Hum Detection and Notch Filtering

`restore_hum.py`

The restoration does not simply assume that the interference is exactly 50 Hz.

It:

1. searches the 20–300 Hz region
2. finds narrow spectral peaks
3. checks temporal persistence with an STFT
4. scores candidates using spectral contrast, persistence, and stability
5. chooses the strongest hum candidate
6. applies a second-order IIR notch filter

The notch filter uses:

```text
Q = 30
```

The detected hum frequency in the experiment was:

```text
49.987793 Hz
```

### Measured Results

| Metric | Before restoration | After restoration |
|---|---:|---:|
| RMSE | 0.212098 | 0.008910 |
| Correlation with reference | 0.562117 | 0.998103 |

Measured hum attenuation:

```text
-34.948225 dB
```

The large reduction in RMSE and increase in correlation show that this degradation was highly recoverable with a correctly targeted filter.

Run:

```bash
python code/restore_hum.py
```

## Restoration 2 — Regularized Inverse Dereverberation

`restore_dereverb.py`

Reverberation is modelled as convolution with a room impulse response:

```text
y(t) = x(t) * h(t)
```

The restoration uses a regularized inverse in the frequency domain:

```text
G(f) = H*(f) / (|H(f)|^2 + λ)
```

with:

```text
λ = 0.001
```

Regularization prevents unstable amplification near frequencies where the room response is small.

The restoration deliberately uses an estimated/assumed impulse response rather than a perfectly matching inverse. In the synthetic experiment, the diffuse-tail magnitude used by the degradation and the restoration model is not identical, making the restoration imperfect.

### Measured Results

| Metric | Degraded | Restored |
|---|---:|---:|
| RMSE vs reference | 0.169834 | 0.112117 |
| Correlation with reference | 0.189163 | 0.954548 |
| RMS | 0.120491 | 0.034561 |

Reference RMS:

```text
0.144632
```

The restoration strongly improves correlation but also lowers the signal level and introduces audible background artifacts, demonstrating the trade-off between inverse filtering and model mismatch.

Run:

```bash
python code/restore_dereverb.py
```

## Breaking the Pipeline

`test_wrong_restoration.py`

The dereverberation method is intentionally applied to the clipped signal even though clipping is not a convolutional room-response degradation.

Observed effects include:

- ringing / echo-like artifacts
- increased background noise
- spectral distortion
- unnatural amplitude excursions
- loss of natural speech dynamics
- the original clipping damage remaining unrecovered

The experiment demonstrates that a restoration method is only meaningful when its assumed degradation model actually matches the signal.

Run:

```bash
python code/test_wrong_restoration.py
```

## Cleaner Sound Does Not Always Mean Higher Fidelity

`q7_pitch_test.py`

A low-frequency rumble is first added to the reference recording. The noisy signal is then cleaned using an aggressive sixth-order 250 Hz high-pass filter.

The result sounds cleaner, but the filter also removes low-frequency speech information.

Pitch is tracked using `librosa.pyin`.

### Results

| Metric | Reference | Processed |
|---|---:|---:|
| Median F0 | 110.48 Hz | 111.76 Hz |
| Pitch confidence | 0.1866 | 0.1112 |

Although the median estimated pitch stays similar, the pitch confidence decreases substantially. The audio can therefore sound cleaner to a human listener while becoming less informative to a downstream speech-processing system.

Run:

```bash
python code/q7_pitch_test.py
```

## Full Run Order

From `01_audio_restoration/`:

```bash
python code/baseline.py
python code/add_hum.py
python code/bandpass.py
python code/clipping.py
python code/reverb.py
python code/lossy_compress.py

python code/restore_hum.py
python code/restore_dereverb.py
python code/test_wrong_restoration.py
python code/q7_pitch_test.py
```

`analyze_audio.py` is imported by the other scripts and does not need to be run separately.

## Main Takeaways

- Different views of the same signal expose different information.
- Additive narrowband interference can often be removed with little damage when its frequency is identifiable.
- Filtering and clipping can destroy information that no later restoration can exactly reconstruct.
- Reverberation can be partially inverted when the room response is known or estimated, but inverse filtering is sensitive to mismatch.
- Applying the wrong restoration model can make a signal worse.
- Perceptual cleanliness and machine-usable fidelity are not the same thing.
