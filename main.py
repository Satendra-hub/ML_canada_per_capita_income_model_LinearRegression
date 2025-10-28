# main.py
# This script performs linear regression on Canada's per capita income data to predict future values.
# It loads data, trains a model, evaluates accuracy, makes predictions, visualizes results, and saves the model.

import pandas as pd  # Library for data manipulation and analysis, used to read CSV and handle DataFrames
import matplotlib.pyplot as plt  # Library for plotting graphs, used to visualize data and predictions
from sklearn.linear_model import LinearRegression  # Machine learning model for linear regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Metrics to evaluate model performance
from joblib import dump  # Library to save and load machine learning models

def load_data(filepath):
    """Load and prepare dataset."""
    # Reads the CSV file into a DataFrame, sets column names, and returns features (X) and target (y)
    df = pd.read_csv(filepath)
    df.columns = ['year', 'income']  # Rename columns for clarity
    return df[['year']], df['income']  # X is year column, y is income column

def train_model(X, y):
    """Train linear regression model."""
    # Creates and fits a LinearRegression model to the data
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, year):
    """Predict income for a given year."""
    # Uses the trained model to predict income for a specific year, ensuring feature names match
    return model.predict(pd.DataFrame([[year]], columns=['year']))[0]

def visualize(X, y, model, year, predicted_income):
    """Plot regression line and prediction."""
    # Creates a scatter plot of actual data, plots the regression line, and marks the prediction point
    plt.scatter(X, y, color='blue', label='Actual Data')  # Plot actual data points
    plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Plot fitted line
    plt.scatter(year, predicted_income, color='green', label=f'{year} Prediction')  # Plot prediction
    plt.xlabel("Year")  # X-axis label
    plt.ylabel("Per Capita Income (US$)")  # Y-axis label
    plt.title("Canada Per Capita Income Prediction")  # Plot title
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid for better readability
    plt.show()  # Display the plot

if __name__ == "__main__":
    # Main execution block: loads data, trains model, evaluates, predicts, visualizes, and saves

    # Load and prepare data from CSV file
    X, y = load_data("canada_per_capita_income.csv")

    # Train the linear regression model on the data
    model = train_model(X, y)

    # Evaluate model accuracy using metrics
    y_pred = model.predict(X)  # Get predictions for training data
    r2 = r2_score(y, y_pred)  # R² measures how well the model fits the data (0-1, higher is better)
    mse = mean_squared_error(y, y_pred)  # MSE measures average squared error (lower is better)
    mae = mean_absolute_error(y, y_pred)  # MAE measures average absolute error (lower is better)

    print(f"\n📊 Model Accuracy Metrics:")
    print(f"R² Score: {r2:.4f}")  # Print R² with 4 decimal places
    print(f"Mean Squared Error (MSE): {mse:,.2f}")  # Print MSE with commas and 2 decimals
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")  # Print MAE with commas and 2 decimals

    # Predict income for the year 2020
    year = 2020
    predicted_income = predict(model, year)
    print(f"\n💰 Predicted Per Capita Income in {year}: ${predicted_income:,.2f}")

    # Visualize the data, regression line, and prediction
    visualize(X, y, model, year, predicted_income)

    # Save the trained model to a file for future use
    dump(model, 'predict_income.joblib')

