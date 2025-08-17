import pandas as pd

def perform_correlation_analysis(input_file, target_column):
    df = pd.read_csv(input_file)

    # Ensure 'Timestamp' is a datetime object and set as index for consistency with cleaning
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

    # Create a lagged feature for Outdoor_Temperature (using the 15-hour lag for humidity)
    # This is to ensure we are correlating with the relevant lagged feature if it's dropped elsewhere
    lag_in_30min_intervals = 30 # 15 hours
    df['Outdoor_Temperature_Lagged'] = df['Outdoor_Temperature'].shift(lag_in_30min_intervals)
    df.dropna(inplace=True)

    # Drop original Outdoor_Temperature as we are using the lagged version for correlation
    if 'Outdoor_Temperature' in df.columns:
        df = df.drop(columns=['Outdoor_Temperature'])

    # Select only numerical columns for correlation calculation
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Calculate correlation with the target_column (Humidity)
    correlations = df[numerical_cols].corr()[target_column].sort_values(ascending=False)

    print(f"Correlation of other parameters with {target_column}:")
    print(correlations)

if __name__ == "__main__":
    input_csv_file = 'cleaned_30min_data.csv'
    target_temperature_column = 'Humidity' # Changed to Humidity
    perform_correlation_analysis(input_csv_file, target_temperature_column) 