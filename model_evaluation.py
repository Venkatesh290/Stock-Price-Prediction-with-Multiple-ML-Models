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
