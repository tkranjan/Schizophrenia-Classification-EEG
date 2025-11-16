# app_streamlit.py
import streamlit as st
import tempfile, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---- CONFIG ----
MODEL_PATH = "./processed/non_subject_fold10_best_1763145887.h5"        # <- update to your saved model path
MAX_UPLOAD_MB = 200                 # file size guard
DEFAULT_THRESHOLD = 0.5
# ----------------

# Import your real helper functions (from your notebook)
# They must implement EXACT same preprocessing used in training
# Create preproc.py with read_raw_edf_file, preprocess_raw, epoch_raw

from preproc import read_raw_edf_file, preprocess_raw, epoch_raw

st.set_page_config(page_title="EEG Schizophrenia Demo", layout="wide")
st.title("EEG Schizophrenia Detection")

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
agg_method = st.sidebar.selectbox("Aggregation method", ["mean_prob", "majority_vote"])
use_ica = st.sidebar.checkbox("Use ICA in preprocessing (slow)", value=False)
epoch_secs = st.sidebar.number_input("Epoch seconds", value=25, min_value=1, max_value=60)

uploaded = st.file_uploader("Upload single EDF file", type=["edf","EDF","zip"])
st.info("Model loads on first run. Keep the model file path correct in MODEL_PATH.")

@st.cache_resource
def load_eeg_model(model_path):
    return load_model(model_path)

if uploaded is None:
    st.info("Upload an EDF file to run prediction.")
else:
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error("File too large. Reduce file size or use smaller recordings.")
    else:
        # Save temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
        try:
            tmp.write(uploaded.read())
            tmp.flush()
            st.success("File uploaded and saved temporarily.")

            # 1) Read
            raw = read_raw_edf_file(tmp.name)
            st.write("Raw info: channels:", len(raw.ch_names), "sfreq:", raw.info['sfreq'])

            # 2) Preprocess
            raw_clean = preprocess_raw(raw, use_ica=use_ica)
            epochs, channels_used = epoch_raw(raw_clean, epoch_secs=epoch_secs)
            n_segments = epochs.shape[0]
            if n_segments == 0:
                st.error("No usable segments found. Try different preprocessing settings or a longer recording.")
            else:
                st.success(f"Extracted {n_segments} epochs (each {epoch_secs}s). Channels used: {channels_used}")

                # 3) Load model and predict
                model = load_eeg_model(MODEL_PATH)
                X_input = np.transpose(epochs, (0,2,1)).astype(np.float32)  # (n, timesteps, channels)
                probs = model.predict(X_input, batch_size=8).ravel()
                seg_labels = (probs >= threshold).astype(int)

                # 4) Aggregate
                if agg_method == "mean_prob":
                    mean_prob = float(np.mean(probs))
                    final_label = int(mean_prob >= threshold)
                    detail = {
                        "n_segments": int(len(probs)),
                        "mean_prob": mean_prob,
                        "std_prob": float(np.std(probs)),
                        "fraction_positive": float(np.mean(seg_labels))
                    }
                else:
                    npos = int(seg_labels.sum())
                    final_label = 1 if npos > (len(probs)/2) else 0
                    detail = {"n_segments": int(len(probs)), "n_positive": npos, "fraction_positive": npos/len(probs)}

                label_text = "Schizophrenia (model positive)" if final_label==1 else "Control (model negative)"
                st.header("Prediction result")
                st.metric("Final label", label_text)
                st.metric("Mean probability", f"{detail.get('mean_prob',0):.4f}")
                st.metric("Fraction positive segments", f"{detail.get('fraction_positive',0):.4f}")

                # 5) Visuals
                fig, axs = plt.subplots(1,2, figsize=(10,3))
                axs[0].hist(probs, bins=20)
                axs[0].set_title("Segment probability distribution")
                axs[1].plot(probs, marker='o')
                axs[1].axhline(threshold, color='red', linestyle='--')
                axs[1].set_title("Per-segment probabilities")
                st.pyplot(fig)

                # waveform example
                st.subheader("Example epoch waveform (first epoch, channel 0)")
                fig2, ax2 = plt.subplots(1,1, figsize=(10,2))
                ax2.plot(epochs[0,0,:])
                ax2.set_title(f"Channel {channels_used[0] if channels_used else 0}")
                st.pyplot(fig2)

                # download csv
                df = pd.DataFrame({"segment_index": np.arange(len(probs)), "prob": probs, "pred": seg_labels})
                st.download_button("Download per-segment CSV", df.to_csv(index=False).encode('utf-8'), "subject_results.csv")
        finally:
            try:
                tmp.close()
                os.unlink(tmp.name)
            except:
                pass
