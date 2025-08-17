import pandas as pd

def convert_hourly_to_daily(input_file, output_file):
    df = pd.read_csv(input_file)

    # Assuming 'Timestamp' is the column containing datetime information
    # Convert 'Timestamp' column to datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')

    # Set 'Timestamp' as the index
    df.set_index('Timestamp', inplace=True)

    # Resample the data to 30-minute frequency and calculate the mean for numerical columns
    # You might want to adjust the aggregation method (e.g., sum, max, min) based on your data.
    daily_df = df.resample('30Min').mean()

    # Save the 30-minute data to a new CSV file
    daily_df.to_csv(output_file)
    print(f"Hourly data converted to 30-minute data and saved to {output_file}")

if __name__ == "__main__":
    input_csv_file = '5.csv'
    output_csv_file = '30min_data.csv'
    convert_hourly_to_daily(input_csv_file, output_csv_file) 