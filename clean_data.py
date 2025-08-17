import pandas as pd

def clean_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Fill missing values in 'Temperature' column using forward fill
    df['Temperature'] = df['Temperature'].ffill()

    # Fill missing values in 'Outdoor_Temperature' column using forward fill
    df['Outdoor_Temperature'] = df['Outdoor_Temperature'].ffill()

    # Fill missing values in 'Humidity', 'Wind_Speed', 'Horizontal_Inlet1_Damper_Position', and 'Vertical_Inlet1_Position' columns using forward fill
    df['Humidity'] = df['Humidity'].ffill()
    df['Wind_Speed'] = df['Wind_Speed'].ffill()
    df['Horizontal_Inlet1_Damper_Position'] = df['Horizontal_Inlet1_Damper_Position'].ffill()
    df['Vertical_Inlet1_Position'] = df['Vertical_Inlet1_Position'].ffill()

    # Fill missing values in remaining columns using forward fill
    df['Outlet1_1_Damper_Position'] = df['Outlet1_1_Damper_Position'].ffill()
    df['Stepless_Gear1'] = df['Stepless_Gear1'].ffill()
    df['Stepless_Variable_Fan1'] = df['Stepless_Variable_Fan1'].ffill()
    df['Vertical_Multi_Gear1'] = df['Vertical_Multi_Gear1'].ffill()
    df['Vertical_Multi_Gear2'] = df['Vertical_Multi_Gear2'].ffill()
    df['Vertical_Multi_Gear3'] = df['Vertical_Multi_Gear3'].ffill()
    df['Vertical_Cooling_Demand_Percent'] = df['Vertical_Cooling_Demand_Percent'].ffill()

    # Fill any remaining missing values (e.g., at the start of the series) using backward fill
    df['Animal'] = df['Animal'].bfill()
    df['Outlet1_1_Damper_Position'] = df['Outlet1_1_Damper_Position'].bfill()
    df['Vertical_Cooling_Demand_Percent'] = df['Vertical_Cooling_Demand_Percent'].bfill()

    # Drop irrelevant columns that are entirely (or mostly) NaN
    columns_to_drop = ['Outlet1_1_Damper_Position', 'Vertical_Cooling_Demand_Percent']
    df = df.drop(columns=columns_to_drop)

    print("Missing values after cleaning:")
    print(df.isnull().sum())
    print("Remaining columns after dropping:")
    print(df.columns.tolist())

    # Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Data cleaned and saved to {output_file}")

if __name__ == "__main__":
    input_csv_file = '30min_data.csv'
    output_csv_file = 'cleaned_30min_data.csv'
    clean_data(input_csv_file, output_csv_file) 