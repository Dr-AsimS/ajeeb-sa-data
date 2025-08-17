import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def build_and_evaluate_model(input_file, target_column):
    df = pd.read_csv(input_file)

    # Drop Timestamp as it's not a direct feature for linear regression
    # We'll keep Animal for now, assuming it's a categorical feature that might be encoded later
    # but for now, we'll drop it if it's not purely numeric or constant.
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    # Identify numerical columns for features
    # Exclude the target column itself
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    X = df[numerical_cols]
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5 # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance for {target_column} Prediction:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Display coefficients to understand feature importance
    print("\nFeature Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.2f}")

    # Plot: Actual vs Predicted Temperature
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2) # y=x line
    plt.title('Actual vs. Predicted Temperature')
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.savefig('actual_vs_predicted_temperature.png')
    plt.close()

if __name__ == "__main__":
    input_csv_file = 'cleaned_30min_data.csv'
    target_temperature_column = 'Temperature'
    build_and_evaluate_model(input_csv_file, target_temperature_column) 