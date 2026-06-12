# 🏡 Smart Property Valuation System

An AI-powered House Price Prediction web application built using **TensorFlow**, **Artificial Neural Networks (ANN)**, and **Streamlit**. The application predicts the estimated price of a house based on various property features such as area, number of bedrooms, bathrooms, parking spaces, furnishing status, and other amenities.

---

## 🚀 Features

* Predict house prices using a trained ANN model
* Interactive and user-friendly Streamlit interface
* Real-time predictions
* One-Hot Encoding for furnishing status
* Feature Scaling using StandardScaler
* Target Variable Scaling for improved regression performance
* Responsive and modern UI
* Supports multiple house attributes for accurate estimation

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Pickle

---

## 📊 Dataset Features

The model uses the following property attributes:

* Area (sq ft)
* Bedrooms
* Bathrooms
* Stories
* Main Road Access
* Guest Room
* Basement
* Hot Water Heating
* Air Conditioning
* Parking Capacity
* Preferred Area
* Furnishing Status

### Target Variable

* House Price

---

## 🧠 Machine Learning Workflow

1. Data Preprocessing

   * Binary Encoding (Yes/No Features)
   * One-Hot Encoding (Furnishing Status)
   * Feature Scaling
   * Target Variable Scaling

2. Model Development

   * Artificial Neural Network (ANN)
   * Dense Hidden Layers with ReLU Activation
   * Mean Absolute Error (MAE) Loss Function
   * Early Stopping
   * TensorBoard Monitoring

3. Model Deployment

   * Streamlit Web Application
   * Real-Time House Price Predictions

##
