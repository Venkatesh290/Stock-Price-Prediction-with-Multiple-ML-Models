# Stock Price Prediction with Multiple ML Models


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol='AAPL', period='2y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.rf_model = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {self.symbol}...")
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(period=self.period)
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def create_technical_indicators(self):
        """Create technical indicators as features"""
        df = self.data.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        
        self.data = df.dropna()
        return self.data
    
    def prepare_lstm_data(self, sequence_length=60):
        """Prepare data for LSTM model"""
        # Use only Close price for LSTM
        data = self.data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test
    
    def prepare_rf_data(self):
        """Prepare data for Random Forest model"""
        features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'MA_20', 
                   'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 
                   'Volatility', 'Price_Change', 'Volume_MA']
        
        df = self.data[features + ['Close']].dropna()
        
        X = df[features]
        y = df['Close']
        
        return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_models(self):
        """Train both LSTM and Random Forest models"""
        print("Training models...")
        
        # Train LSTM
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = self.prepare_lstm_data()
        self.lstm_model = self.build_lstm_model((X_train_lstm.shape[1], 1))
        
        history = self.lstm_model.fit(
            X_train_lstm, y_train_lstm,
            epochs=50, batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # Train Random Forest
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = self.prepare_rf_data()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train_rf, y_train_rf)
        
        # Store test data for evaluation
        self.X_test_lstm, self.y_test_lstm = X_test_lstm, y_test_lstm
        self.X_test_rf, self.y_test_rf = X_test_rf, y_test_rf
        
        return history
    
    def evaluate_models(self):
        """Evaluate both models"""
        print("Evaluating models...")
        
        # LSTM predictions
        lstm_pred_scaled = self.lstm_model.predict(self.X_test_lstm)
        lstm_pred = self.scaler.inverse_transform(lstm_pred_scaled)
        y_test_lstm_actual = self.scaler.inverse_transform(self.y_test_lstm.reshape(-1, 1))
        
        # Random Forest predictions
        rf_pred = self.rf_model.predict(self.X_test_rf)
        
        # Calculate metrics
        lstm_mse = mean_squared_error(y_test_lstm_actual, lstm_pred)
        lstm_rmse = np.sqrt(lstm_mse)
        lstm_mae = mean_absolute_error(y_test_lstm_actual, lstm_pred)
        lstm_r2 = r2_score(y_test_lstm_actual, lstm_pred)
        
        rf_mse = mean_squared_error(self.y_test_rf, rf_pred)
        rf_rmse = np.sqrt(rf_mse)
        rf_mae = mean_absolute_error(self.y_test_rf, rf_pred)
        rf_r2 = r2_score(self.y_test_rf, rf_pred)
        
        results = {
            'LSTM': {
                'RMSE': lstm_rmse,
                'MAE': lstm_mae,
                'R²': lstm_r2,
                'predictions': lstm_pred.flatten(),
                'actual': y_test_lstm_actual.flatten()
            },
            'Random Forest': {
                'RMSE': rf_rmse,
                'MAE': rf_mae,
                'R²': rf_r2,
                'predictions': rf_pred,
                'actual': self.y_test_rf.values
            }
        }
        
        # Print results
        print("\n=== Model Evaluation Results ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  R²: {metrics['R²']:.4f}")
        
        return results
    
    def plot_predictions(self, results):
        """Plot prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # LSTM predictions plot
        axes[0, 0].plot(results['LSTM']['actual'][-100:], label='Actual', alpha=0.7)
        axes[0, 0].plot(results['LSTM']['predictions'][-100:], label='LSTM Prediction', alpha=0.7)
        axes[0, 0].set_title('LSTM Model - Last 100 Predictions')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Random Forest predictions plot
        axes[0, 1].plot(results['Random Forest']['actual'][-100:], label='Actual', alpha=0.7)
        axes[0, 1].plot(results['Random Forest']['predictions'][-100:], label='RF Prediction', alpha=0.7)
        axes[0, 1].set_title('Random Forest Model - Last 100 Predictions')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Feature importance for Random Forest
        feature_names = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'MA_20', 
                        'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 
                        'Volatility', 'Price_Change', 'Volume_MA']
        
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        axes[1, 0].bar(range(10), importances[indices])
        axes[1, 0].set_title('Top 10 Feature Importances (Random Forest)')
        axes[1, 0].set_xticks(range(10))
        axes[1, 0].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        
        # Stock price with technical indicators
        recent_data = self.data.tail(100)
        axes[1, 1].plot(recent_data.index, recent_data['Close'], label='Close Price')
        axes[1, 1].plot(recent_data.index, recent_data['MA_20'], label='MA 20', alpha=0.7)
        axes[1, 1].plot(recent_data.index, recent_data['BB_upper'], label='BB Upper', alpha=0.5)
        axes[1, 1].plot(recent_data.index, recent_data['BB_lower'], label='BB Lower', alpha=0.5)
        axes[1, 1].set_title('Stock Price with Technical Indicators')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create an interactive Plotly dashboard"""
        recent_data = self.data.tail(200)
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=recent_data.index,
            open=recent_data['Open'],
            high=recent_data['High'],
            low=recent_data['Low'],
            close=recent_data['Close'],
            name='Price'
        )])
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['MA_20'],
            name='MA 20',
            line=dict(color='orange', width=2)
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['BB_upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash'),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['BB_lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            opacity=0.5,
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ))
        
        fig.update_layout(
            title=f'{self.symbol} Stock Analysis Dashboard',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        fig.show()
        
        return fig

