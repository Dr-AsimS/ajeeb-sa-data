import pandas as pd

def perform_correlation_analysis(input_file, target_column):
    df = pd.read_csv(input_file)

    # Convert 'Timestamp' to datetime and set as index if not already done
    # This is important for time-series operations, but not strictly for correlation here
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # df.set_index('Timestamp', inplace=True)

    # Select only numerical columns for correlation calculation
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Calculate correlation with the target_column (Temperature)
    correlations = df[numerical_cols].corr()[target_column].sort_values(ascending=False)

    print(f"Correlation of other parameters with {target_column}:")
    print(correlations)

if __name__ == "__main__":
    input_csv_file = 'cleaned_30min_data.csv'
    target_temperature_column = 'Temperature'
    perform_correlation_analysis(input_csv_file, target_temperature_column) 