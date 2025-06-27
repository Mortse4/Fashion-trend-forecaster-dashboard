import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import plotly.graph_objects as go
from plotly.io import to_html

class TrendForecaster:
    def __init__(self, window=6, epochs=100):
        self.window = window
        self.model = None
        self.epochs = epochs
        self.scaler = MinMaxScaler()

    def sliding_window(self, trend_df, frequency_col="ScaledFrequency"):
        data = trend_df.sort_values(["Year"])
        features = data[["Year", frequency_col]].values.astype("float32")
        self.input_shape = features.shape[1]

        x, y = [], []
        for i in range(len(features) - self.window):
            x.append(features[i:i + self.window])
            y.append(data[frequency_col].iloc[i + self.window])

        x, y = np.array(x, dtype="float32"), np.array(y, dtype="float32")
        print("Shape of x before returning from sliding_window:", x.shape)
        print("x range:", np.min(x), np.max(x))
        print("y range:", np.min(y), np.max(y))
        return x, y

    def build_forecasting_model(self, input_shape):
        """
        Constructs a hybrid LSTM + Transformer model for trend forecasting.

        Args:
            input_shape (Tuple[int, int]): Shape of the input sequences.

        Returns:
            tf.keras.Model: Compiled forecasting model.
        """

        inputs = layers.Input(shape=input_shape)
        lstm_out = layers.LSTM(64, return_sequences=True)(inputs)
        lstm_out = layers.Dropout(0.1)(lstm_out)

        attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
        attn_output = layers.Dropout(0.1)(attn_output)
        attn_output = layers.LayerNormalization(epsilon=1e-6)(lstm_out + attn_output)

        ffn = layers.Dense(64, activation='relu')(attn_output)
        ffn = layers.Dense(64)(ffn)
        ffn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output + ffn)

        x = layers.GlobalAveragePooling1D()(ffn_output)
        outputs = layers.Dense(1, activation='linear')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return self.model

    def split_data(self, x, y, train_size=0.7, val_size=0.15):
        """
        Splits data into training, validation, and test sets.

        Args:
            x (np.ndarray): Input sequences.
            y (np.ndarray): Target values.
            train_size (float): Proportion of training data.
            val_size (float): Proportion of validation data.

        Returns:
            Tuple: Split data (train_x, train_y, val_x, val_y, test_x, test_y).
        """
        total_size = len(x)
        train_end = int(total_size * train_size)
        val_end = train_end + int(total_size * val_size)

        train_x, train_y = x[:train_end], y[:train_end]
        val_x, val_y = x[train_end:val_end], y[train_end:val_end]
        test_x, test_y = x[val_end:], y[val_end:]

        return train_x, train_y, val_x, val_y, test_x, test_y

    def train_model(self, train_x, train_y, val_x, val_y, epochs=100, batch_size=16):
        """
        Trains the forecasting model.

        Args:
            train_x, train_y: Training data.
            val_x, val_y: Validation data.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.

        Returns:
            History object from model training.
        """
        print("Starting model training...")
        history = self.model.fit(
            train_x, train_y,
            validation_data=(val_x, val_y),
            epochs=epochs,
            batch_size=batch_size,
        )
        return history

    def evaluate_model(self, test_x, test_y):
        return self.model.evaluate(test_x, test_y)

    def predict(self, test_x):
        return self.model.predict(test_x)
    
    

    def forecast_future(self, model, last_known_data, future_years=3):
        """
        Iteratively forecasts future values based on last known input sequence.

        Args:
            model (tf.keras.Model): Trained model.
            last_known_data (np.ndarray): Last known window of input data.
            future_years (int): Number of steps to forecast.

        Returns:
            np.ndarray: Forecasted future values.
        """
        predictions = []
        current_input = last_known_data.copy()

        for _ in range(future_years):
            next_pred = model.predict(current_input, verbose=0)[0][0]
            predictions.append(next_pred)

            last_year = current_input[0, -1, 0]
            next_year = last_year + 1
            next_row = np.array([next_year, next_pred])
            next_row = next_row.astype("float32")
            next_row = next_row[np.newaxis, np.newaxis, :]
            current_input = np.concatenate((current_input[:, 1:, :], next_row), axis=1)

        return np.array(predictions)


    def plot_log_difference_forecast(self, top_clusters_df):
        """
        Trains and visualizes log-difference forecasts for each top cluster.

        Args:
            top_clusters_df (pd.DataFrame): DataFrame containing cluster frequencies over time.

        Returns:
            Tuple: Dictionary of models and predictions for each cluster.
        """
        models = {}
        predictions = {}

        for cluster_id, cluster_data in top_clusters_df.groupby("Cluster"):
            # Aggregate the data by year and sum the frequencies
            cluster_data = cluster_data.groupby("Year").agg({
                "Frequency": "sum",
                "OriginalYear": "first"
            }).reset_index().sort_values("Year")

            # Apply log transformation and calculate the log differences
            cluster_data["LogFreq"] = np.log(cluster_data["Frequency"] + 1)
            cluster_data["LogDiff"] = cluster_data["LogFreq"].diff()

            logdiff_data = cluster_data.dropna()
            x, y = self.sliding_window(logdiff_data, frequency_col="LogDiff")

            # Skip clusters that don't have enough data for the sliding window
            if len(x) < self.window:
                print(f"Skipping Cluster {cluster_id} (not enough data for log diff window)")
                continue

            # Split the data into training, validation, and test sets
            train_x, train_y, val_x, val_y, test_x, test_y = self.split_data(x, y)

            # Build and train the forecasting model
            model = self.build_forecasting_model(input_shape=(self.window, self.input_shape))
            history = self.train_model(train_x, train_y, val_x, val_y, epochs=self.epochs, batch_size=32)

            # Predict over all input windows (for comparison with actual)
            all_predictions = model.predict(x).flatten()
            original_years = logdiff_data["Year"].values
            prediction_years = original_years[self.window:]

            # Forecast future log differences
            last_input = logdiff_data[["Year", "LogDiff"]].values[-self.window:]
            last_input = np.reshape(last_input, (1, self.window, self.input_shape)).astype("float32")
            future_logdiffs = self.forecast_future(model, last_input, future_years=3)

            last_logfreq = cluster_data["LogFreq"].iloc[-1]
            forecasted_logfreqs = [last_logfreq + np.sum(future_logdiffs[:i + 1]) for i in range(len(future_logdiffs))]
            forecasted_freqs = [np.exp(lf) - 1 for lf in forecasted_logfreqs]


            last_predicted_year = prediction_years[-1]
            # Forecast future (to start from 1 and increment by 0.1)
            future_years = [last_predicted_year + i * 0.1 for i in range(len(future_logdiffs))]

            # Plotting
            fig = go.Figure()

            # Plot actual log differences
            fig.add_trace(go.Scatter(
                x=logdiff_data["Year"].values,
                y=logdiff_data["LogDiff"],
                mode='lines+markers',
                name='Actual Log Difference',
                line=dict(color='#2A2729')
            ))

            # Plot predicted historical log differences
            fig.add_trace(go.Scatter(
                x=prediction_years,
                y=all_predictions,
                mode='lines+markers',
                name='Predicted Log Difference (Historical)',
                line=dict(color='paleturquoise')
            ))

            # Plot forecasted future log differences
            fig.add_trace(go.Scatter(
                x=future_years,
                y=future_logdiffs,
                mode='lines+markers',
                name='Forecasted Log Difference (Future)',
                line=dict(dash='dot', color='saddlebrown')
            ))

            # Update layout for the plot
            fig.update_layout(
                title=f"Log-Diff Trend Forecast for Cluster {cluster_id}",
                xaxis_title="Scaled Year",
                yaxis_title="Popularity",
                template="plotly_white",
                paper_bgcolor="#FAF6F7",
                plot_bgcolor="#FFFFFF"
            )

            fig.show()

            # Store the model and predictions for the cluster
            models[cluster_id] = model
            predictions[cluster_id] = {
                "Years": prediction_years,
                "HistoricalPredictions": all_predictions,
                "FutureYears": future_years,
                "FutureForecast": future_logdiffs
            }

        return models, predictions



    def process_top_clusters(self, top_clusters_df):
        """
        Trains forecasting models on top clusters and visualizes raw frequency forecasts.
        This was not used in the report, this was the initial grpah until we settled on the 
        log difference graph with scaled year and frequency plotted. But can be ran to see 
        the difference between the two graphs for each cluster.
        Use runVogue2.py to see results outside of the dashboard
        Args:
            top_clusters_df (pd.DataFrame): DataFrame containing frequency data per cluster.

        Returns:
            Tuple: Dictionary of trained models and predicted future frequencies.
        """
        models = {}
        predictions = {}

        for cluster_id, cluster_data in top_clusters_df.groupby("Cluster"):
            cluster_data = cluster_data.groupby("Year").agg({
                "Frequency": "sum",
                "ScaledFrequency": "sum",
                "OriginalYear": "first"
            }).reset_index().sort_values("Year")

            if len(cluster_data) < self.window:
                print(f"Skipping Cluster {cluster_id} due to insufficient data.")
                continue

            x, y = self.sliding_window(cluster_data, frequency_col="ScaledFrequency")
            train_x, train_y, val_x, val_y, test_x, test_y = self.split_data(x, y)
            model = self.build_forecasting_model(input_shape=(self.window, self.input_shape))
            print(f"Training Cluster {cluster_id}: x shape {x.shape}, y shape {y.shape}")
            history = self.train_model(train_x, train_y, val_x, val_y, epochs=self.epochs, batch_size=32)
            models[cluster_id] = model

            last_known_data = cluster_data[["Year", "ScaledFrequency"]].values[-self.window:]
            last_known_data = np.reshape(last_known_data, (1, self.window, self.input_shape)).astype("float32")
            future_pred = self.forecast_future(model, last_known_data, future_years=3)
            predictions[cluster_id] = future_pred

            all_x = np.concatenate([train_x, val_x, test_x])
            all_pred = model.predict(all_x, verbose=0).flatten()
            all_years_original = cluster_data["OriginalYear"].iloc[self.window - 1 : self.window - 1 + len(all_pred)].values
            all_frequencies = cluster_data["Frequency"].iloc[self.window - 1 : self.window - 1 + len(all_pred)].values

            scaler_input = np.column_stack((all_years_original, all_frequencies))
            self.scaler.fit(scaler_input)

            combined_pred = np.column_stack((all_years_original, all_pred))
            inverse_pred = self.scaler.inverse_transform(combined_pred)
            all_pred_inversed = inverse_pred[:, 1]

            last_train_year = all_years_original[-1]
            future_years_original = np.array([last_train_year + i + 1 for i in range(len(future_pred))])
            future_combined = np.column_stack((future_years_original, future_pred))
            future_inverse = self.scaler.inverse_transform(future_combined)
            future_pred_inversed = future_inverse[:, 1]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cluster_data["OriginalYear"], y=cluster_data["Frequency"], mode='lines+markers', name='Actual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=all_years_original, y=all_pred_inversed, mode='lines', name='Model Prediction (Historical)', line=dict(dash='dash', color='orange')))
            fig.add_trace(go.Scatter(x=future_years_original, y=future_pred_inversed, mode='lines+markers', name='Forecasted', line=dict(dash='dot', color='green')))
            fig.update_layout(title=f"Trend Forecast for Cluster {cluster_id}", xaxis_title="Year", yaxis_title="Frequency", template="plotly_white")
            fig.show()

        return models, predictions
