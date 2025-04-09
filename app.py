
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from streamlit_lottie import st_lottie
from rapidfuzz import process
import json
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="🎵 MatchMyMelody", page_icon="🎶", layout="wide")

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
.chat-bubble {
    background-color: #eeeeff;
    padding: 1em;
    border-radius: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Lottie Animation
# -------------------------------
def load_lottiefile(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

animation = load_lottiefile("music-animation.json")

# -------------------------------
# User Authentication
# -------------------------------
USER_CREDENTIALS = {
    "adyasha": "melody123",
    "admin": "admin123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def show_login():
    st.markdown("<h2 style='text-align: center;'>🔐 Login to MatchMyMelody</h2>", unsafe_allow_html=True)
    if animation:
        st_lottie(animation, height=200)
    
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")
    
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.success(f"✅ Welcome, {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Try again.")

if not st.session_state.logged_in:
    show_login()
    st.stop()

# -------------------------------
# App Title
# -------------------------------
st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🎵 MatchMyMelody</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover Hindi songs with a similar vibe 🎶</p>", unsafe_allow_html=True)
if animation:
    st_lottie(animation, height=240)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Hindi_songs.csv")
df.fillna(0, inplace=True)

def convert_to_seconds(time_str):
    try:
        m, s = map(int, str(time_str).split(':'))
        return m * 60 + s
    except:
        return 0

df['duration'] = df['duration'].apply(convert_to_seconds)

def assign_mood(row):
    valence = row['Valence']
    energy = row['energy']
    danceability = row['danceability']
    
    if valence >= 0.7 and energy >= 0.6:
        return "Happy"
    elif energy >= 0.75 and danceability >= 0.65:
        return "Energetic"
    elif valence >= 0.6 and energy < 0.5:
        return "Romantic"
    elif valence < 0.4 and energy < 0.5:
        return "Sad"
    elif valence < 0.4 and energy >= 0.5:
        return "Heartbreak"
    elif 0.4 <= valence <= 0.7 and energy < 0.6:
        return "Calm"
    else:
        return "Calm"

df['Mood'] = df.apply(assign_mood, axis=1)

# -------------------------------
# Feature Sliders
# -------------------------------
st.markdown("### 🎛️ Customize what matters to you:")

energy_weight = st.slider("⚡ Energy", 0.0, 2.0, 1.0)
dance_weight = st.slider("💃 Danceability", 0.0, 2.0, 1.0)
acoustic_weight = st.slider("🎻 Acousticness", 0.0, 2.0, 1.0)
valence_weight = st.slider("😊 Valence (positivity)", 0.0, 2.0, 1.0)
liveness_weight = st.slider("🎤 Liveness", 0.0, 2.0, 1.0)
duration_weight = st.slider("⏱️ Duration", 0.0, 2.0, 1.0)

df_feat = df[['duration', 'danceability', 'acousticness', 'energy', 'liveness', 'Valence']].copy()
df_feat['duration'] *= duration_weight
df_feat['danceability'] *= dance_weight
df_feat['acousticness'] *= acoustic_weight
df_feat['energy'] *= energy_weight
df_feat['liveness'] *= liveness_weight
df_feat['Valence'] *= valence_weight

# -------------------------------
# Normalize and Fit k-NN
# -------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_feat)
knn = NearestNeighbors(n_neighbors=11, metric='cosine')
knn.fit(df_scaled)

# -------------------------------
# Helper Functions
# -------------------------------
def find_index_by_song_name(name):
    names = df['song_name'].tolist()
    best = process.extractOne(name, names, score_cutoff=60)
    if best:
        return df[df['song_name'] == best[0]].index[0]
    return None

def recommend_songs(index, n_recs=5):
    dist, indices = knn.kneighbors([df_scaled[index]])
    recs = []
    for i in range(1, n_recs + 1):
        idx = indices[0][i]
        row = df.iloc[idx]
        recs.append({
            "🎵 Song": row['song_name'],
            "🎤 Artist": row.get('singer', 'Unknown'),
            "📅 Released": row.get('released_date', 'N/A'),
            "⚡ Energy": round(row.get('energy', 0), 2),
            "😊 Mood": row['Mood']
        })
    return pd.DataFrame(recs)

# -------------------------------
# Recommendation Section
# -------------------------------
st.markdown("---")
mode = st.radio("Choose input method:", ["🎼 Dropdown", "💬 Assistant"])

mood_option = st.selectbox("🎭 Optional: Filter by Mood", ["All"] + sorted(df['Mood'].unique()))
selected_mood = None if mood_option == "All" else mood_option

if mode == "🎼 Dropdown":
    song_list = sorted(df['song_name'].dropna().unique())
    default_idx = song_list.index("Tum Hi Ho") if "Tum Hi Ho" in song_list else 0

    with st.form("dropdown_form"):
        song_name = st.selectbox("🎵 Song Name", song_list, index=default_idx)
        submit = st.form_submit_button("✨ Recommend")

    if submit:
        idx = find_index_by_song_name(song_name)
        if idx is not None:
            st.success(f"✅ Songs similar to: {song_name} (Mood: {mood_option})")
            results = recommend_songs(idx)
            if selected_mood:
                results = results[results['😊 Mood'].str.lower() == selected_mood.lower()]
            if not results.empty:
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("⚠️ No matching songs found for the selected mood.")
        else:
            st.error("❌ Song not found.")

elif mode == "💬 Assistant":
    query = st.text_input("🗣️ What would you like to hear?")
    if st.button("Ask"):
        st.markdown(f"<div class='chat-bubble'>🧑‍💻 You: {query}</div>", unsafe_allow_html=True)

        moods = []
        lowered = query.lower()

        if any(word in lowered for word in ["happy", "joy", "smile", "dance", "energetic", "cheerful"]):
            moods.append("Happy")
        if any(word in lowered for word in ["sad", "cry", "breakup", "heart", "alone"]):
            moods.append("Sad")
        if any(word in lowered for word in ["calm", "chill", "relax", "soothing", "peace", "slow"]):
            moods.append("Calm")
        if any(word in lowered for word in ["party", "workout", "pump", "jump", "beat"]):
            moods.append("Energetic")
        if any(word in lowered for word in ["love", "romance", "romantic", "date", "sweet"]):
            moods.append("Romantic")
        if any(word in lowered for word in ["heartbreak", "lost", "tears", "miss", "pain"]):
            moods.append("Heartbreak")

        if moods:
            mood_text = ", ".join(moods)
            st.markdown(f"<div class='chat-bubble'>🤖 Assistant: Detected mood(s) - {mood_text}</div>", unsafe_allow_html=True)
            mood_df = df[df['Mood'].str.lower().isin([m.lower() for m in moods])]
            if not mood_df.empty:
                idx = mood_df.sample(1).index[0]
                results = recommend_songs(idx)
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("⚠️ No songs found for the detected mood(s). Showing a random mix.")
                st.dataframe(recommend_songs(df.sample(1).index[0]), use_container_width=True)
        else:
            st.warning("🤔 Couldn’t detect a clear mood. Showing a random recommendation.")
            st.dataframe(recommend_songs(df.sample(1).index[0]), use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("🎧 Enjoy your personalized melody match!")

 