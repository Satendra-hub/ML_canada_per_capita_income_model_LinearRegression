# Predict canada's per capita income in year 2020. There is an exercise folder here on github at same level as this notebook,    download that and you will find canada_per_capita_income.csv file. Using this build a regression model and predict the per capita income fo canadian citizens in year 2020

# 🇨🇦 Canada Per Capita Income Prediction (1970–2020)

This project uses **Linear Regression** to predict Canada's per capita income based on historical data.

## 📊 Dataset

- Source: `canada_per_capita_income.csv`
- Columns: `year`, `income`

## 🔧 Features

- Loads and prepares data
- Trains a regression model
- Predicts income for a future year (e.g., 2020)
- Visualizes regression line and prediction
- Saves model using Joblib

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py

# Output or result--

📊 Model Accuracy Metrics:
R² Score: 0.8909
Mean Squared Error (MSE): 15,462,739.06
Mean Absolute Error (MAE): 3,088.87

💰 Predicted Per Capita Income in 2020: $41,288.69

