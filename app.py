import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import matplotlib.pyplot as plt
import librosa.display

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(
    page_title="Music Genre Classifier 🎧",
    page_icon="🎶",
    layout="centered"
)

# ------------------ LOAD CSS ------------------
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
qda = joblib.load("models/qda_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ------------------ HEADER ------------------
st.markdown("""
<div class="header">
    <h1>🎶 Music Genre Classifier</h1>
    <p>Upload a song & let AI predict the genre</p>
</div>
""", unsafe_allow_html=True)

# ------------------ FILE UPLOADER ------------------
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

# ------------------ MAIN LOGIC ------------------
if audio_file:

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(audio_file.read())
        file_path = temp.name  # NOW DEFINED

    # Play audio preview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.audio(file_path)
    st.markdown("</div>", unsafe_allow_html=True)

    # Load audio data (NOW SAFE TO USE file_path!)
    y, sr = librosa.load(file_path, mono=True)

    # -------------------- VISUALIZATION --------------------
    st.markdown("### 📊 Audio Visualizations")

    tab1, tab2 = st.tabs(["🎚 Waveform", "📡 Spectrogram"])

    # ---- WAVEFORM ----
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#1db954")
        ax.set_facecolor("black")
        st.pyplot(fig)

    # ---- SPECTROGRAM ----
    with tab2:
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig2, ax2 = plt.subplots(figsize=(10, 3))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax2)
        ax2.set_facecolor("black")
        st.pyplot(fig2)


    # ------------------ FEATURE EXTRACTION ------------------
    def extract_features(file_path):
        y, sr = librosa.load(file_path, mono=True)

        length = len(y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_cross = librosa.feature.zero_crossing_rate(y)
        harmony = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        features = [
            length,
            np.mean(chroma), np.var(chroma),
            np.mean(rms), np.var(rms),
            np.mean(spectral_centroid), np.var(spectral_centroid),
            np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zero_cross), np.var(zero_cross),
            np.mean(harmony), np.var(harmony),
            np.mean(percussive), np.var(percussive),
            tempo
        ]

        for i in range(20):
            features.append(np.mean(mfcc[i]))
            features.append(np.var(mfcc[i]))

        return np.array(features).reshape(1, -1)


    # ------------------ PREDICT BUTTON ------------------
    if st.button("🎧 Predict Genre"):
        with st.spinner("Analyzing song..."):
            features = extract_features(file_path)
            scaled = scaler.transform(features)
            prediction = qda.predict(scaled)[0]

        st.markdown(f"""
        <div class="result-card">
            <h2>🎵 Predicted Genre</h2>
            <div class="genre-pill">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("⬆ Upload a music file to continue.")
