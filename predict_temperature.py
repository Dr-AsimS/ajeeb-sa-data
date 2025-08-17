import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model # Import Model for functional API
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Flatten, MultiHeadAttention, LayerNormalization, Dropout # Add Transformer layers
from tensorflow.keras.optimizers import Adam # Explicitly import Adam
import tensorflow as tf # Import tensorflow for positional embedding
from xgboost import XGBRegressor # Import XGBoost

# Helper function to create sequences for LSTMs/Transformers
def create_sequences(data, look_back, target_column_loc):
    X, y, timestamps = [], [], []
    for i in range(len(data) - look_back):
        X.append(data.iloc[i:(i + look_back), :].values)
        y.append(data.iloc[i + look_back, target_column_loc])
        timestamps.append(data.index[i + look_back])
    return np.array(X), np.array(y), np.array(timestamps)

# Positional Embedding for Transformer
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=output_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-2]
        positions = tf.range(start=0, limit=length, delta=1)
        return inputs + self.position_embedding(positions)

# Transformer Encoder Block
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None): # Added training=None
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_and_evaluate_model(input_file, target_column):
    df = pd.read_csv(input_file)

    # Ensure 'Timestamp' is a datetime object and set as index
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

    # Create a lagged feature for Outdoor_Temperature
    lag_in_30min_intervals = 30  # 15 hours for humidity
    df['Outdoor_Temperature_Lagged'] = df['Outdoor_Temperature'].shift(lag_in_30min_intervals)

    # Drop original Outdoor_Temperature as we are using the lagged version
    if 'Outdoor_Temperature' in df.columns:
        df = df.drop(columns=['Outdoor_Temperature'])

    # Drop rows with NaNs introduced by lagging (usually at the beginning)
    df.dropna(inplace=True)

    # Identify numerical columns for features
    feature_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_column in feature_cols:
        feature_cols.remove(target_column)

    # Add Prophet features if they exist in the dataframe
    if 'trend' in df.columns and 'daily' in df.columns and 'weekly' in df.columns:
        feature_cols.extend(['trend', 'daily', 'weekly'])
        # Ensure Prophet features are numeric after merging (sometimes they might be objects)
        df['trend'] = pd.to_numeric(df['trend'], errors='coerce')
        df['daily'] = pd.to_numeric(df['daily'], errors='coerce')
        df['weekly'] = pd.to_numeric(df['weekly'], errors='coerce')
        df.dropna(subset=['trend', 'daily', 'weekly'], inplace=True) # Drop rows if conversion created NaNs

    # Add new time-based features if they exist in the dataframe
    if 'hour' in df.columns and 'dayofweek' in df.columns and 'dayofyear' in df.columns and 'month' in df.columns:
        feature_cols.extend(['hour', 'dayofweek', 'dayofyear', 'month'])

    # For XGBoost, directly use X and y
    # We need to preserve original index for timestamps if it's datetime
    if df.index.name == 'Timestamp':
        df = df.reset_index()

    X = df[feature_cols]
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost Regressor model
    model = XGBRegressor(objective='reg:squarederror', random_state=42) # Changed to XGBoost
    print(f"\nTraining XGBoost model for {target_column} prediction (with all features & 15-hour lagged Outdoor Temp)...") # Updated print
    model.fit(X_train.values, y_train.values) # Convert to numpy arrays
    print("Training complete.")

    # Make predictions
    y_pred = model.predict(X_test.values) # Convert to numpy array

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance for {target_column} Prediction (XGBoost with all features & 15-hour lagged Outdoor Temp):") # Updated print
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Display Feature Importances for XGBoost
    print("\nFeature Importances:")
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_importances.to_string())

    # Plot: Actual vs Predicted Humidity over Time (for XGBoost with all features)
    test_indices = X_test.index
    full_timestamps = df['Timestamp']
    X_test_timestamps = full_timestamps.loc[test_indices]

    results_df = pd.DataFrame({
        'Timestamp': X_test_timestamps,
        'Actual_{}'.format(target_column): y_test,
        'Predicted_{}'.format(target_column): y_pred
    })
    results_df = results_df.sort_values(by='Timestamp')

    plt.figure(figsize=(15, 7))
    plt.plot(results_df['Timestamp'], results_df['Actual_{}'.format(target_column)], label=f'Actual {target_column}', color='blue')
    plt.plot(results_df['Timestamp'], results_df['Predicted_{}'.format(target_column)], label=f'Predicted {target_column}', color='red', linestyle='--')
    plt.title(f'Actual vs. Predicted {target_column} Over Time (XGBoost with All Features & 15-hour Lagged Outdoor Temp)') # Updated title
    plt.xlabel('Timestamp')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_{target_column.lower()}_timeseries_xgboost_all_features_15hr_lag.png') # New filename
    plt.close()

if __name__ == "__main__":
    input_csv_file = 'cleaned_30min_data_all_features.csv' # Changed input file
    target_temperature_column = 'Humidity' # Target is Humidity
    build_and_evaluate_model(input_csv_file, target_temperature_column) 