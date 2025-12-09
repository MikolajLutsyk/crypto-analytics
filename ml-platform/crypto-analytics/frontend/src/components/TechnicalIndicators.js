import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const TechnicalIndicators = ({ data }) => {
  if (!data) return <div className="loading">Loading technical indicators...</div>;

  // Подготавливаем данные для RSI графика
  const rsiData = data.timestamps.map((timestamp, index) => ({
    timestamp: new Date(timestamp).toLocaleDateString(),
    rsi: data.indicators.rsi_14[index],
    overbought: 70,
    oversold: 30
  }));

  // Подготавливаем данные для MACD графика
  const macdData = data.timestamps.map((timestamp, index) => ({
    timestamp: new Date(timestamp).toLocaleDateString(),
    macd: data.indicators.macd[index],
    signal: data.indicators.macd_signal[index]
  }));

  const formatTooltip = (value, name) => {
    return [value?.toFixed(4) || 'N/A', name];
  };

  return (
    <div className="technical-container">
      <h3>Technical Indicators</h3>
      
      <div className="indicators-grid">
        {/* RSI Indicator */}
        <div className="indicator-chart">
          <h4>RSI (14-period)</h4>
          <p className="indicator-description">
            Relative Strength Index - measures momentum. Above 70 = overbought, Below 30 = oversold.
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={rsiData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="timestamp" angle={-45} textAnchor="end" height={50} />
              <YAxis domain={[0, 100]} />
              <Tooltip formatter={formatTooltip} />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="rsi" 
                stroke="#8884d8" 
                name="RSI" 
                strokeWidth={2}
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="overbought" 
                stroke="#ff4d4f" 
                name="Overbought (70)" 
                strokeDasharray="3 3"
                strokeWidth={1}
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="oversold" 
                stroke="#52c41a" 
                name="Oversold (30)" 
                strokeDasharray="3 3"
                strokeWidth={1}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* MACD Indicator */}
        <div className="indicator-chart">
          <h4>MACD</h4>
          <p className="indicator-description">
            Moving Average Convergence Divergence - trend-following momentum indicator.
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={macdData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="timestamp" angle={-45} textAnchor="end" height={50} />
              <YAxis />
              <Tooltip formatter={formatTooltip} />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="macd" 
                stroke="#1890ff" 
                name="MACD" 
                strokeWidth={2}
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="signal" 
                stroke="#ff7a45" 
                name="Signal Line" 
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Indicator Summary */}
      <div className="indicator-summary">
        <h4>Current Indicator Values:</h4>
        <div className="indicator-values">
          <div className="indicator-value">
            <span>RSI:</span>
            <strong>{data.indicators.rsi_14[data.indicators.rsi_14.length - 1]?.toFixed(2) || 'N/A'}</strong>
          </div>
          <div className="indicator-value">
            <span>MACD:</span>
            <strong>{data.indicators.macd[data.indicators.macd.length - 1]?.toFixed(4) || 'N/A'}</strong>
          </div>
          <div className="indicator-value">
            <span>MACD Signal:</span>
            <strong>{data.indicators.macd_signal[data.indicators.macd_signal.length - 1]?.toFixed(4) || 'N/A'}</strong>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TechnicalIndicators;