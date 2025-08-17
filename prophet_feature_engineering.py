import pandas as pd
from prophet import Prophet

def extract_prophet_features(input_file, output_file, target_column='Humidity'):
    df = pd.read_csv(input_file)

    # Ensure 'Timestamp' is a datetime object, do not set as index yet
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    else:
        print("Warning: 'Timestamp' column not found or not in expected format. Ensure it's present and correct.")
        return

    # --- Add basic time-based features ---
    df['hour'] = df['Timestamp'].dt.hour
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['dayofyear'] = df['Timestamp'].dt.dayofyear
    df['month'] = df['Timestamp'].dt.month

    # Prepare data for Prophet (ds, y format)
    prophet_df = df[['Timestamp', target_column]].rename(columns={'Timestamp': 'ds', target_column: 'y'})

    # Initialize and fit Prophet model
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)

    # Create a DataFrame for future predictions (which is just the historical data points)
    future = model.make_future_dataframe(periods=0, freq='30Min')
    
    # Predict to get components
    forecast = model.predict(future)

    # Merge these components back to the original dataframe
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    df_with_features = df.merge(forecast[['ds', 'trend', 'daily', 'weekly']], left_on='Timestamp', right_on='ds', how='left')
    
    # Drop the 'ds' column from the merged dataframe as it's a duplicate of Timestamp
    df_with_features.drop(columns=['ds'], inplace=True)

    # Save the new DataFrame with extracted features, without index
    df_with_features.to_csv(output_file, index=False) # index=False to prevent Unnamed: 0
    print(f"Data with Prophet features and time features saved to {output_file}")

if __name__ == "__main__":
    input_csv_file = 'cleaned_30min_data.csv'
    output_csv_file = 'cleaned_30min_data_all_features.csv' # New output file
    extract_prophet_features(input_csv_file, output_csv_file) 