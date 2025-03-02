                                                                   Airline Profitability Prediction

📌 Project Overview

This project aims to predict airline profitability using a machine learning model trained on historical flight performance data. By leveraging various operational features such as flight delays, aircraft utilization, load factor, fleet availability, maintenance downtime, fuel efficiency, revenue, and operating costs, the model provides insights to optimize airline operations.

🚀 Key Features & Technologies

Machine Learning Model: Trained using Random Forest Regressor.

Preprocessing Pipeline: Handles categorical encoding, feature scaling, and missing data imputation.

Deployment: Interactive Streamlit web application.

Tools & Libraries: Python, Scikit-learn, Pandas, NumPy, Joblib, and Streamlit.

⚙️ Setup Instructions

Follow these steps to set up and run the project locally:


Run the Streamlit Application

streamlit run app.py

🏢 Department-Specific Focus

📊 Approach
1.Data Cleaning & Preprocessing
Handled missing values in numerical columns (mean imputation) and categorical columns (mode imputation).
Scaled numerical features using StandardScaler.
Encoded categorical variables (Flight Number, Route Type) using Label Encoding.
2. Feature Engineering
Created new features:
departure_delay = actual_departure_hour - scheduled_departure_hour
route_domestic & route_international (one-hot encoded).
Placeholder extra features for model consistency.
Ensured consistency: Matched feature order with the trained model.
3. Model Training
Used Random Forest Regressor for prediction.
Evaluated multiple models, selecting the best performing one.
4. Evaluation Metrics
Root Mean Squared Error (RMSE): Measures model error magnitude.
Mean Absolute Error (MAE): Captures average prediction error.
R² Score: Indicates model performance.
Metric	Value
RMSE	945.32
MAE	680.45
R² Score	0.86


🔗 GitHub Repository:https://github.com/joshin-joseph-au5/MASAI-HACKATHON.git                                 
📂 LARGE FILES & Presentation: https://drive.google.com/drive/folders/1WI60wN3pslxpRhVYeJKUALluVu3xBhsq?usp=drive_link                               

