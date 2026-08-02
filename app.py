import streamlit as st
import pandas as pd
import joblib

# ===========================
# Load Model and Dataset
# ===========================

model = joblib.load("model/car_price_model.pkl")
df = pd.read_csv("dataset/car_data.csv")

# ===========================
# Page
# ===========================

st.set_page_config(
    page_title="Smart Car Price Advisor",
    page_icon="🚗"
)

st.title("🚗 Smart Car Price Advisor")
st.write("Predict the fair price of a used car.")

# ===========================
# Inputs
# ===========================

brand = st.selectbox(
    "Brand",
    sorted(df["Brand"].unique())
)

models = sorted(
    df[df["Brand"] == brand]["Model"].unique()
)

model_name = st.selectbox(
    "Model",
    models
)

year = st.number_input(
    "Year",
    min_value=int(df["Year"].min()),
    max_value=int(df["Year"].max()),
    value=int(df["Year"].median())
)

engine_size = st.number_input(
    "Engine Size",
    min_value=float(df["Engine_Size"].min()),
    max_value=float(df["Engine_Size"].max()),
    value=float(df["Engine_Size"].median())
)

fuel = st.selectbox(
    "Fuel Type",
    sorted(df["Fuel_Type"].unique())
)

transmission = st.selectbox(
    "Transmission",
    sorted(df["Transmission"].unique())
)

mileage = st.number_input(
    "Mileage",
    min_value=float(df["Mileage"].min()),
    max_value=float(df["Mileage"].max()),
    value=float(df["Mileage"].median())
)

doors = st.number_input(
    "Doors",
    min_value=int(df["Doors"].min()),
    max_value=int(df["Doors"].max()),
    value=int(df["Doors"].median())
)

owner_count = st.number_input(
    "Previous Owners",
    min_value=int(df["Owner_Count"].min()),
    max_value=int(df["Owner_Count"].max()),
    value=int(df["Owner_Count"].median())
)

advertisement_price = st.number_input(
    "Advertisement Price",
    min_value=0.0,
    value=float(df["Price"].median())
)

# ===========================
# Prediction
# ===========================

if st.button("Predict Price"):

    car = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Year": [year],
        "Engine_Size": [engine_size],
        "Fuel_Type": [fuel],
        "Transmission": [transmission],
        "Mileage": [mileage],
        "Doors": [doors],
        "Owner_Count": [owner_count]
    })

    predicted_price = model.predict(car)[0]

    difference = advertisement_price - predicted_price

    percentage = (difference / predicted_price) * 100

    st.subheader("Prediction Result")

    st.success(
        f"Estimated Price: {predicted_price:,.0f}"
    )

    st.write(
        f"Advertisement Price: {advertisement_price:,.0f}"
    )

    st.write(
        f"Difference: {difference:,.0f}"
    )

    st.write(
        f"Difference Percentage: {percentage:.2f}%"
    )

    if percentage < -5:
        st.success("✅ Good Deal")
        st.write("The advertisement price is lower than the predicted price.")

    elif percentage > 5:
        st.warning("⚠️ Overpriced")
        st.write("The advertisement price is higher than the predicted price.")

    else:
        st.info("👍 Fair Price")
        st.write("The advertisement price is close to the predicted price.")

# ===========================
# Sidebar
# ===========================

st.sidebar.title("Project Information")
st.sidebar.write("Machine Learning Regression Project")
st.sidebar.write("Models: Linear Regression, Decision Tree, Random Forest")
st.sidebar.write("Built with Streamlit")
