import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_temperature_lag(input_file):
    df = pd.read_csv(input_file)

    # Ensure 'Timestamp' is a datetime object and set as index
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

    # Select the two columns for analysis
    indoor_series = df['Humidity'] # Changed to Humidity
    outdoor_series = df['Outdoor_Temperature'] # Retain Outdoor_Temperature for cross-correlation

    # Remove any remaining NaNs in these specific series before cross-correlation
    indoor_series = indoor_series.dropna()
    outdoor_series = outdoor_series.dropna()

    # Calculate cross-correlation for a range of lags
    max_lag = 48 # 48 * 30 minutes = 24 hours
    lags = range(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:
        if lag >= 0:
            corr = indoor_series.corr(outdoor_series.shift(lag))
        else:
            corr = indoor_series.shift(-lag).corr(outdoor_series)
        correlations.append(corr)

    # Convert lags to hours for plotting
    lags_hours = [l * 0.5 for l in lags]

    # Plotting the cross-correlation
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=lags_hours, y=correlations)
    plt.title('Cross-Correlation between Indoor Humidity and Outdoor Temperature')
    plt.xlabel('Lag (Hours)')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8) # Zero lag line
    plt.savefig('humidity_lag_analysis.png') # Changed filename
    plt.close()

    # Find the lag with the highest correlation
    max_corr_index = np.argmax(correlations)
    optimal_lag_30min = lags[max_corr_index]
    optimal_lag_hours = optimal_lag_30min * 0.5
    max_correlation_value = correlations[max_corr_index]

    print(f"\nLag Analysis Results for Humidity vs Outdoor Temperature:") # Updated print statement
    print(f"Maximum correlation (Pearsons): {max_correlation_value:.2f}")
    print(f"Optimal Lag: {optimal_lag_30min} (30-minute intervals)")
    print(f"Optimal Lag: {optimal_lag_hours} (hours)")
    print("\nInterpretation: A positive lag means Outdoor_Temperature leads Indoor Humidity.")
    print("Plot saved: 'humidity_lag_analysis.png'") # Updated plot filename

    # Plot: Indoor Humidity vs Lagged Outdoor Temperature (Time Series)
    plt.figure(figsize=(15, 7))
    start_date = df.index.min()
    end_date = start_date + pd.Timedelta(days=5) # Plotting first 5 days for clarity

    plt.plot(df.loc[start_date:end_date, 'Humidity'], label='Indoor Humidity', color='green')
    plt.plot(df.loc[start_date:end_date, 'Outdoor_Temperature'].shift(optimal_lag_30min),
             label=f'Outdoor Temperature (shifted by {optimal_lag_hours} hours)', color='red', linestyle='--')

    plt.title(f'Indoor Humidity vs. Outdoor Temperature (Shifted by {optimal_lag_hours} Hours)') # Updated title
    plt.xlabel('Timestamp')
    plt.ylabel('Humidity') # Changed ylabel
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lagged_humidity_time_series.png') # Changed filename
    plt.close()

if __name__ == "__main__":
    input_csv_file = 'cleaned_30min_data.csv'
    analyze_temperature_lag(input_csv_file) 