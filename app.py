import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from streamlit_lottie import st_lottie
import json
import os

# Set page config FIRST
st.set_page_config(page_title="🎵 MatchMyMelody", page_icon="🎶", layout="wide")

# Optional Styling
st.markdown("""
    <style>
    input {
        border-radius: 10px;
        padding: 10px;
    }
    button[kind="primary"] {
        background-color: #6a0dad;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load Lottie animation from JSON file
def load_lottiefile(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        st.warning(f"⚠️ Animation file '{filepath}' not found.")
        return None

# Title and animation
st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🎵 MatchMyMelody</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover Hindi songs with a similar vibe 🎶</p>", unsafe_allow_html=True)
animation = load_lottiefile("music-animation.json")
if animation:
    st_lottie(animation, height=240)

# Load dataset
df = pd.read_csv("Hindi_songs.csv")
df.fillna(0, inplace=True)

# Convert duration (mm:ss) to seconds
def convert_to_seconds(time_str):
    try:
        minutes, seconds = map(int, str(time_str).split(':'))
        return minutes * 60 + seconds
    except:
        return 0

df['duration'] = df['duration'].apply(convert_to_seconds)

# Select features and scale
features = ['duration', 'danceability', 'acousticness', 'energy', 'liveness', 'Valence']
df_features = df[features]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# Fit k-NN model
knn = NearestNeighbors(n_neighbors=11, metric='cosine')
knn.fit(df_scaled)

# Helper functions
def find_index_by_song_name(song_name):
    matches = df[df['song_name'].str.lower() == song_name.lower()]
    return matches.index[0] if not matches.empty else None

def recommend_songs(index, n_recommendations=5):
    distances, indices = knn.kneighbors([df_scaled[index]])
    recs = []
    for i in range(1, n_recommendations + 1):
        idx = indices[0][i]
        song = df.iloc[idx]
        recs.append({
            "🎵 Song": song['song_name'],
            "🎤 Artist": song.get('singer', 'Unknown'),
            "🌐 Language": song.get('language', 'N/A'),
            "📅 Released": song.get('released_date', 'N/A'),
            "⚡ Energy": round(song.get('energy', 0), 2)
        })
    return pd.DataFrame(recs)

# Input Section with Dropdown and Styled Form
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🔍 Enter or Select a Hindi Song You Like</h3>", unsafe_allow_html=True)

song_list = sorted(df['song_name'].dropna().unique())
default_index = song_list.index("Tum Hi Ho") if "Tum Hi Ho" in song_list else 0

with st.form("recommendation_form", clear_on_submit=False):
    song_name = st.selectbox(
        "", song_list, index=default_index, label_visibility="collapsed"
    )
    submitted = st.form_submit_button("✨ Recommend Similar Songs")

# Trigger Recommendation
if song_name and submitted:
    index = find_index_by_song_name(song_name)
    if index is not None:
        st.success(f"✅ Songs similar to: {song_name.title()}")
        st.dataframe(recommend_songs(index), use_container_width=True)
    else:
        st.error("❌ Song not found in dataset. Try another name.")

st.markdown("---")
st.caption("🎧 Enjoy your personalized melody match!")
