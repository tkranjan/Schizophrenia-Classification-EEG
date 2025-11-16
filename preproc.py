import mne
import numpy as np
from scipy.signal import resample

# -----------------------------
# READ RAW EDF  (from notebook)
# -----------------------------
def read_raw_edf_file(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    return raw

# -----------------------------
# PREPROCESS RAW (exactly like notebook)
# -----------------------------
def preprocess_raw(raw,
                   target_sfreq=250,
                   notch_freq=50,
                   l_freq=0.5,
                   h_freq=40,
                   use_ica=False):
    # Resample
    raw = raw.copy().resample(target_sfreq)
    
    # Notch filter
    raw.notch_filter(notch_freq)
    
    # Band-pass filter
    raw.filter(l_freq, h_freq)
    
    # Re-reference
    raw.set_eeg_reference('average')

    # OPTIONAL ICA (if you enabled it during training)
    if use_ica:
        ica = mne.preprocessing.ICA(n_components=15, random_state=97)
        ica.fit(raw)
        raw = ica.apply(raw)
    
    return raw

# -----------------------------
# EPOCHING (exactly like training)
# -----------------------------
def epoch_raw(raw, epoch_secs=25):
    sfreq = raw.info['sfreq']
    n_samples = int(epoch_secs * sfreq)

    data = raw.get_data()  # (n_channels, time_samples)
    total = data.shape[1]
    
    segments = []
    idx = 0
    while idx + n_samples <= total:
        seg = data[:, idx : idx + n_samples]
        segments.append(seg)
        idx += n_samples

    segments = np.array(segments)  # (n_segments, n_channels, samples)

    # Per-segment z-scoring
    for i in range(segments.shape[0]):
        segments[i] = (segments[i] - np.mean(segments[i], axis=1, keepdims=True)) / (
            np.std(segments[i], axis=1, keepdims=True) + 1e-8
        )

    return segments, raw.ch_names
