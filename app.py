import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------
# LOAD FILES
# --------------------------------

model = tf.keras.models.load_model("house_model.h5")

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("y_scaler.pkl", "rb") as file:
    y_scaler = pickle.load(file)

with open("onehot_encoder_fur.pkl", "rb") as file:
    onehot_encoder_fur = pickle.load(file)

# --------------------------------
# CUSTOM CSS
# --------------------------------
# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.main{
    background:#0f172a;
}

.block-container{
    padding-top:4rem;
    padding-bottom:2rem;
}

.title{
    text-align:center;
    color:inherit;
    font-size:3.3rem;
    font-weight:800;
    margin-top:15px;
    margin-bottom:10px;
    line-height:1.2;
}

.subtitle{
    text-align:center;
    color:#94a3b8;
    font-size:1.1rem;
    margin-bottom:40px;
}

.stButton button{
    width:100%;
    height:60px;
    border-radius:12px;
    font-size:18px;
    font-weight:700;
}

[data-testid="stVerticalBlock"]{
    gap:1rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# HEADER
# --------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="title">
        🏡 Smart Property Valuation System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-Powered House Price Prediction Using Artificial Neural Networks
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# INPUTS
# --------------------------------

col1,col2 = st.columns(2)

with col1:

    area = st.number_input(
        "📐 Area (sq ft)",
        min_value=500,
        max_value=20000,
        value=5000
    )

    bedrooms = st.slider(
        "🛏 Bedrooms",
        1,10,3
    )

    bathrooms = st.slider(
        "🚿 Bathrooms",
        1,10,2
    )

    stories = st.slider(
        "🏢 Stories",
        1,6,2
    )

    parking = st.slider(
        "🚗 Parking",
        0,5,2
    )

    furnishing = st.selectbox(
        "🪑 Furnishing Status",
        [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]
    )

with col2:

    mainroad = st.selectbox(
        "🛣 Main Road",
        ["yes","no"]
    )

    guestroom = st.selectbox(
        "🛋 Guest Room",
        ["yes","no"]
    )

    basement = st.selectbox(
        "🏚 Basement",
        ["yes","no"]
    )

    hotwaterheating = st.selectbox(
        "♨ Hot Water Heating",
        ["yes","no"]
    )

    airconditioning = st.selectbox(
        "❄ Air Conditioning",
        ["yes","no"]
    )

    prefarea = st.selectbox(
        "📍 Preferred Area",
        ["yes","no"]
    )

# --------------------------------
# PREDICTION
# --------------------------------

if st.button("🔮 Predict House Price"):

    input_df = pd.DataFrame({
        "area":[area],
        "bedrooms":[bedrooms],
        "bathrooms":[bathrooms],
        "stories":[stories],
        "mainroad":[1 if mainroad=="yes" else 0],
        "guestroom":[1 if guestroom=="yes" else 0],
        "basement":[1 if basement=="yes" else 0],
        "hotwaterheating":[1 if hotwaterheating=="yes" else 0],
        "airconditioning":[1 if airconditioning=="yes" else 0],
        "parking":[parking],
        "prefarea":[1 if prefarea=="yes" else 0]
    })

    fur_encoded = onehot_encoder_fur.transform(
        [[furnishing]]
    )

    fur_df = pd.DataFrame(
        fur_encoded,
        columns=onehot_encoder_fur.get_feature_names_out(
            ["furnishingstatus"]
        )
    )

    final_input = pd.concat(
        [input_df, fur_df],
        axis=1
    )

    scaled_input = scaler.transform(
        final_input
    )

    prediction = model.predict(
        scaled_input
    )

    predicted_price = y_scaler.inverse_transform(
        prediction
    )

    st.success(
        f"🏠 Estimated House Price: ₹ {predicted_price[0][0]:,.0f}"
    )

    st.balloons()