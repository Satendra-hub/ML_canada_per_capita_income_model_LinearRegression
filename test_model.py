import pandas as pd
from joblib import load

def test_model_loading():
    """Test loading the saved model and making a prediction."""
    # Load the saved model
    model = load('predict_income.joblib')

    # Load data to get the feature names (not used further, but ensures consistency)

    # Test prediction for 2025
    year = 2025
    predicted_income = model.predict(pd.DataFrame([[year]], columns=['year']))[0]
    print(f"Loaded model prediction for {year}: ${predicted_income:,.2f}")

    # Verify the model coefficients
    print(f"Model coefficient: {model.coef_[0]:.2f}")
    print(f"Model intercept: {model.intercept_:.2f}")

if __name__ == "__main__":
    test_model_loading()
