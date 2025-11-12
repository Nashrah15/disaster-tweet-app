# app_streamlit.py
import streamlit as st
import joblib, os, io
import pandas as pd
import numpy as np
import re
from PIL import Image
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from streamlit_folium import st_folium
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

st.set_page_config(layout="wide", page_title="Disaster Tweet Classifier — Demo")

# -------------------------
# Helpers
# -------------------------
def clean_text(text, lemmatizer=None, stop=None):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9#\s]", " ", text)
    tokens = [w for w in text.split() if (stop is None or w not in stop)]
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def load_models(tfidf_path="models/tfidf_vectorizer.pkl", model_path="models/disaster_model.pkl"):
    tfidf = None; model = None
    if os.path.exists(tfidf_path) and os.path.exists(model_path):
        tfidf = joblib.load(tfidf_path)
        model = joblib.load(model_path)
    return tfidf, model

# -------------------------
# Sidebar: Load models / dataset
# -------------------------
st.sidebar.title("Setup & Files")
tfidf_path = st.sidebar.text_input("TF-IDF path", "models/tfidf_vectorizer.pkl")
model_path = st.sidebar.text_input("Classifier path", "models/disaster_model.pkl")
if st.sidebar.button("Load models"):
    tfidf, model = load_models(tfidf_path, model_path)
    if model is None:
        st.sidebar.error("Model files not found. Place .pkl files in models/ or set correct paths.")
    else:
        st.sidebar.success("Models loaded.")
else:
    tfidf = None; model = None

uploaded_csv = st.sidebar.file_uploader("Upload tweets CSV (optional)", type=["csv"])
df = None
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.sidebar.write("CSV loaded:", df.shape)

# -------------------------
# Page layout
# -------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.header("Tweet text tester")
    user_tweet = st.text_area("Enter a tweet to classify", height=120)
    if st.button("Classify tweet"):
        if tfidf is None or model is None:
            st.warning("Load models first from the sidebar.")
        else:
            # simple cleaning (no heavy lemmatizer here)
            ct = clean_text(user_tweet)
            vec = tfidf.transform([ct])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0].max()
            st.markdown(f"**Prediction:** {'REAL' if pred==1 else 'NOT REAL'}  |  **Confidence:** {prob:.2f}")
    st.markdown("#### Bulk test: Upload CSV and run (optional)")
    if df is not None and st.button("Predict CSV text column"):
        if 'text' not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            if tfidf is None or model is None:
                st.warning("Load models first.")
            else:
                df['clean_text'] = df['text'].astype(str).apply(clean_text)
                X = tfidf.transform(df['clean_text'])
                preds = model.predict(X)
                probs = model.predict_proba(X)[:,1]
                df['pred'] = preds
                df['prob'] = probs
                st.write(df[['text','pred','prob']].head(30))
                st.download_button("Download predictions CSV", df.to_csv(index=False).encode('utf-8'), "preds.csv")

with col2:
    st.header("Map & Location search")
    st.write("If you uploaded a CSV with a 'text' column, we can extract city names and map them.")
    if df is None:
        st.info("Upload a tweets CSV in the sidebar to enable mapping.")
    else:
        # Extract city names using a simple heuristic: hashtags and capitalized words fallback.
        from geotext import GeoText
        df['text'] = df['text'].astype(str)
        df['city'] = df['text'].apply(lambda t: (GeoText(t).cities[0] if GeoText(t).cities else None))
        st.write(f"Detected {df['city'].notnull().sum()} geonames from {len(df)} tweets.")
        # Allow filtering to predicted real if predictions present
        if 'pred' in df.columns:
            show_real_only = st.checkbox("Show only tweets predicted REAL", value=True)
            plot_df = df[df['pred']==1] if show_real_only else df
        else:
            plot_df = df
        # Geocode unique cities (cache in session_state to avoid repeated calls)
        if 'place_coords' not in st.session_state:
            st.session_state.place_coords = {}
        geolocator = Nominatim(user_agent="st_disaster_demo")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        unique_places = plot_df['city'].dropna().unique().tolist()
        with st.spinner("Geocoding places (this may take a while)..."):
            for p in unique_places:
                if p not in st.session_state.place_coords:
                    try:
                        loc = geocode(f"{p}, India")
                        if loc:
                            st.session_state.place_coords[p] = (loc.latitude, loc.longitude)
                        else:
                            st.session_state.place_coords[p] = None
                    except:
                        st.session_state.place_coords[p] = None
        # build map
        start_lat, start_lon = 20.5937, 78.9629
        m = folium.Map(location=[start_lat, start_lon], zoom_start=5)
        from folium.plugins import MarkerCluster
        mc = MarkerCluster().add_to(m)
        for _, r in plot_df.dropna(subset=['city']).iterrows():
            coords = st.session_state.place_coords.get(r['city'])
            if coords:
                folium.Marker(location=[coords[0], coords[1]],
                              popup=f"{r['city']}: {r['text'][:140]}",
                              icon=folium.Icon(color='red' if r.get('pred',1)==1 else 'blue')).add_to(mc)
        st_data = st_folium(m, width=700, height=450)

st.markdown("---")
st.header("Image upload & quick disaster-check")
st.write("Upload an image and the app will run a MobileNet model to look for disaster-related objects (fire, ambulance, smoke, etc.).")
uploaded_img = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded_img is not None:
    image_bytes = uploaded_img.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    st.image(img, caption="Uploaded image", use_column_width=True)
    # prepare and predict
    model_img = mobilenet_v2.MobileNetV2(weights='imagenet')
    img_resized = kimage.load_img(io.BytesIO(image_bytes), target_size=(224,224))
    x = kimage.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_img.predict(x)
    labels = decode_predictions(preds, top=5)[0]
    st.write("Top predictions:", labels)
    labels_join = " ".join([l[1] for l in labels]).lower()
    disaster_keywords = ['fire','flood','smoke','ambulance','rescue','volcano','collapse','flooding','earthquake']
    if any(k in labels_join for k in disaster_keywords):
        st.success("This image likely shows a disaster scene.")
    else:
        st.info("Image does not strongly match disaster-related objects (may be fake/non-disaster).")

st.markdown("### Notes")
st.markdown("- For accurate mapping prefer tweets with explicit city names or 'location' metadata from Twitter API.") 
st.markdown("- MobileNet is a generic object model; for high accuracy fine-tune on a disaster image dataset.")
