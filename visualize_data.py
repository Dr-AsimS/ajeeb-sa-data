import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_relationships(input_file):
    df = pd.read_csv(input_file)

    # Set style for plots
    sns.set_style("whitegrid")

    # Plot: Temperature vs Outdoor_Temperature
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Outdoor_Temperature', y='Temperature', data=df)
    plt.title('Temperature vs. Outdoor Temperature')
    plt.xlabel('Outdoor Temperature')
    plt.ylabel('Temperature')
    plt.savefig('temperature_vs_outdoor_temperature.png')
    plt.close()

    # Plot: Temperature vs Wind_Speed
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Wind_Speed', y='Temperature', data=df)
    plt.title('Temperature vs. Wind Speed')
    plt.xlabel('Wind Speed')
    plt.ylabel('Temperature')
    plt.savefig('temperature_vs_wind_speed.png')
    plt.close()

    # Plot: Actual Humidity over Time
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Humidity'], label='Actual Humidity', color='green')
    plt.title('Actual Humidity Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Humidity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_humidity_timeseries.png')
    plt.close()

    print("Plots saved: 'temperature_vs_outdoor_temperature.png' and 'temperature_vs_wind_speed.png'")

if __name__ == "__main__":
    input_csv_file = 'cleaned_30min_data.csv'
    visualize_relationships(input_csv_file) 