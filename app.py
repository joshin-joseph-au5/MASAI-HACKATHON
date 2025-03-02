import streamlit as st
import numpy as np
import joblib

# Load trained model and preprocessing objects
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define expected feature order
expected_features = [
    "departure_hour", "arrival_hour", "load_factor", "fuel_efficiency",
    "operating_cost", "revenue", "fleet_availability", "maintenance_downtime",
    "departure_delay", "month", "day_of_week", "flight_number_encoded",
    "route_domestic", "route_international",
    "extra_feature_1", "extra_feature_2", "extra_feature_3",
    "extra_feature_4", "extra_feature_5", "extra_feature_6"
]

# Streamlit UI
st.title("Airline Profitability Prediction")
st.sidebar.header("Flight Information")

# Numerical Inputs
departure_hour = st.sidebar.slider("Scheduled Departure Hour", 0, 23, 10)
arrival_hour = st.sidebar.slider("Actual Departure Hour", 0, 23, 12)
load_factor = st.sidebar.slider("Load Factor (%)", 50, 100, 85)
fuel_efficiency = st.sidebar.slider("Fuel Efficiency (ASK)", 0.5, 10.0, 5.0)
operating_cost = st.sidebar.number_input("Operating Cost (USD)", min_value=0.0, value=5000.0)
revenue = st.sidebar.number_input("Revenue (USD)", min_value=0.0, value=10000.0)
fleet_availability = st.sidebar.slider("Fleet Availability (%)", 50, 100, 90)
maintenance_downtime = st.sidebar.number_input("Maintenance Downtime (hrs)", min_value=0.0, value=2.0)

# Derived Features
departure_delay = arrival_hour - departure_hour
month = 6  # Default month
day_of_week = 2  # Default day

# Categorical Inputs
route = st.sidebar.selectbox("Route Type", ["Domestic", "International"])
route_domestic = 1 if route == "Domestic" else 0
route_international = 1 if route == "International" else 0

# Flight Number Encoding
flight_number = st.sidebar.number_input("Flight Number", min_value=1, value=101)
if "Flight Number" in label_encoders:
    if flight_number in label_encoders["Flight Number"].classes_:
        flight_number_encoded = label_encoders["Flight Number"].transform([flight_number])[0]
    else:
        st.warning(f"Flight Number {flight_number} was not seen during training. Using default encoding (0).")
        flight_number_encoded = 0
else:
    flight_number_encoded = 0  

# Placeholder values for missing features
extra_features = [0] * (len(expected_features) - 14)  # Fill with zeros

# Arrange features in the correct order
features = np.array([[
    departure_hour, arrival_hour, load_factor, fuel_efficiency, 
    operating_cost, revenue, fleet_availability, maintenance_downtime,
    departure_delay, month, day_of_week, flight_number_encoded,
    route_domestic, route_international] + extra_features])

# Validate feature count
if features.shape[1] != len(expected_features):
    st.error(f"Feature mismatch! Expected {len(expected_features)} features, but got {features.shape[1]}.")
else:
    # Apply scaling
    features_scaled = scaler.transform(features)
    
    # Prediction
    if st.sidebar.button("Predict Profit"):
        predicted_profit = model.predict(features_scaled)
        st.subheader(f"Predicted Profit: **${predicted_profit[0]:,.2f}**")
