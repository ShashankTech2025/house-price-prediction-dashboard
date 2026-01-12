import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import time

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---------------- Load Model & Data ----------------
model = pickle.load(open("house_model.pkl", "rb"))
df = pd.read_csv("Housing.csv")

# ---------------- Sidebar Settings ----------------
st.sidebar.title("⚙ Dashboard Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# ---------------- Location-wise Backgrounds ----------------
bg_images = {
    "Pune": "https://images.unsplash.com/photo-1568605114967-8130f3a36994",
    "Mumbai": "https://images.unsplash.com/photo-1501594907352-04cda38ebc29",
    "Bangalore": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c"
}

location = st.selectbox("📍 Select City", list(bg_images.keys()))

bg_image = bg_images[location]
theme = "plotly_dark" if dark_mode else "plotly_white"
text_color = "white" if dark_mode else "#111"

# ---------------- Custom CSS ----------------
st.markdown(f"""
<style>
.main {{
    background-image: url("{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.glass {{
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    color: {text_color};
}}

.counter {{
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="glass">
<h2 style="text-align:center;">🏠 House Price Prediction</h2>
<p style="text-align:center;">Smart • Fast • User Friendly</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Animated Counters ----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

def animated_counter(label, value):
    st.markdown(f"<p style='text-align:center;'>{label}</p>", unsafe_allow_html=True)
    placeholder = st.empty()
    for i in range(0, int(value)+1, max(1, int(value/25))):
        placeholder.markdown(f"<div class='counter'>{i}</div>", unsafe_allow_html=True)
        time.sleep(0.02)
    placeholder.markdown(f"<div class='counter'>{int(value)}</div>", unsafe_allow_html=True)

with col1:
    animated_counter("🏘 Total Houses", df.shape[0])

with col2:
    animated_counter("💰 Avg Price", df.price.mean())

with col3:
    animated_counter("📐 Avg Area", df.area.mean())

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction Section ----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("🔢 Enter House Details")

area = st.number_input("📐 Area (sqft)", 300, 10000, step=50)
bedrooms = st.number_input("🛏 Bedrooms", 1, 10)
bathrooms = st.number_input("🚿 Bathrooms", 1, 10)
stories = st.number_input("🏢 Stories", 1, 5)
parking = st.number_input("🚗 Parking", 0, 5)

if st.button("🔮 Predict Price", use_container_width=True):
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class='glass'>
            <h3 style='text-align:center;'>💰 Estimated Price</h3>
            <h1 style='text-align:center;color:#00ffcc;'>₹ {prediction:,.2f}</h1>
            <p style='text-align:center;'>📍 {location}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Market Insights ----------------
st.markdown("""
<div class="glass">
<h3>📊 Market Insights</h3>
</div>
""", unsafe_allow_html=True)

fig = px.scatter(
    df,
    x="area",
    y="price",
    size="bedrooms",
    color="bathrooms",
    template=theme,
    title="Area vs Price"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("""
<div class="glass">
<p style="text-align:center;">📱 Mobile Friendly • ☁ Cloud Ready • 🎓 Resume Project</p>
</div>
""", unsafe_allow_html=True)
