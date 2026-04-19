import streamlit as st
import pandas as pd
from predict import classify_property

st.title("Housing Price Anomaly Detector")

# Basic Inputs
area = st.number_input("Area", min_value=0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
stories = st.number_input("Stories", min_value=0)

mainroad = st.selectbox("Main Road", [1, 0])
guestroom = st.selectbox("Guest Room", [1, 0])
basement = st.selectbox("Basement", [1, 0])
hotwaterheating = st.selectbox("Hot Water Heating", [1, 0])
airconditioning = st.selectbox("Air Conditioning", [1, 0])
parking = st.number_input("Parking", min_value=0)
prefarea = st.selectbox("Preferred Area", [1, 0])

furnishing = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

actual_price = st.number_input("Actual Price", min_value=0)

# Button
if st.button("Predict"):

    # One-hot encoding
    furnished = 1 if furnishing == "furnished" else 0
    semi = 1 if furnishing == "semi-furnished" else 0
    unfurnished = 1 if furnishing == "unfurnished" else 0

    # Create DataFrame
    features = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus_furnished": furnished,
        "furnishingstatus_semi-furnished": semi,
        "furnishingstatus_unfurnished": unfurnished
    }])

    # Call your logic
    pred, gap, pct, label = classify_property(features, actual_price)

    # Output
    st.subheader("Result")
    st.write(f"Predicted Price: ₹{pred:,.0f}")
    st.write(f"Gap: ₹{gap:,.0f}")
    st.write(f"Gap %: {pct:.2f}%")
    st.write(f"Decision: {label}")
