# Example usage
def main():
    # Initialize predictor
    predictor = StockPredictor(symbol='AAPL', period='2y')
    
    # Fetch and prepare data
    predictor.fetch_data()
    predictor.create_technical_indicators()
    
    # Train models
    history = predictor.train_models()
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Create visualizations
    predictor.plot_predictions(results)
    predictor.create_interactive_dashboard()

if __name__ == "__main__":
    main()

# Additional utility functions for extended analysis

def predict_future_prices(predictor, days=30):
    """Predict future stock prices"""
    last_sequence = predictor.data['Close'].values[-60:].reshape(-1, 1)
    last_sequence_scaled = predictor.scaler.transform(last_sequence)
    
    predictions = []
    current_batch = last_sequence_scaled.reshape(1, 60, 1)
    
    for i in range(days):
        next_pred = predictor.lstm_model.predict(current_batch)[0]
        predictions.append(next_pred[0])
        
        # Update batch for next prediction
        current_batch = np.append(current_batch[:, 1:, :], 
                                 next_pred.reshape(1, 1, 1), axis=1)
    
    # Inverse transform predictions
    predictions = predictor.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()

def calculate_trading_signals(data):
    """Generate basic trading signals"""
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0
    
    # Simple moving average crossover strategy
    signals['Signal'][data['MA_5'] > data['MA_20']] = 1
    signals['Signal'][data['MA_5'] < data['MA_20']] = -1
    
    # RSI overbought/oversold
    signals.loc[data['RSI'] > 70, 'Signal'] = -1  # Sell signal
    signals.loc[data['RSI'] < 30, 'Signal'] = 1   # Buy signal
    
    return signals
