import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const TechnicalIndicators = ({ data }) => {
  if (!data) return <div className="loading">Loading technical indicators...</div>;

  // Проверяем структуру данных и адаптируем их
  const hasFullData = data.timestamps && data.indicators;
  const hasTechData = data.rsi && data.macd;
  
  if (!hasFullData && !hasTechData) {
    return <div className="error">Invalid data format received</div>;
  }

  let rsiData, macdData;
  
  if (hasFullData) {
    // Формат из /api/ohlcv/{symbol}
    rsiData = data.timestamps.map((timestamp, index) => ({
      timestamp: new Date(timestamp).toLocaleDateString(),
      rsi: data.indicators.rsi_14[index],
      overbought: 70,
      oversold: 30
    }));

    macdData = data.timestamps.map((timestamp, index) => ({
      timestamp: new Date(timestamp).toLocaleDateString(),
      macd: data.indicators.macd[index],
      signal: data.indicators.macd_signal[index]
    }));
  } else {
    // Формат из /api/technical/indicators
    // Создаем искусственные временные метки, если их нет
    const timestamps = Array.from({ length: data.rsi.length }, (_, i) => 
      new Date(Date.now() - (data.rsi.length - i - 1) * 24 * 60 * 60 * 1000)
    );
    
    rsiData = timestamps.map((timestamp, index) => ({
      timestamp: timestamp.toLocaleDateString(),
      rsi: data.rsi[index],
      overbought: 70,
      oversold: 30
    }));

    macdData = timestamps.map((timestamp, index) => ({
      timestamp: timestamp.toLocaleDateString(),
      macd: data.macd[index],
      signal: data.macd_signal[index]
    }));
  }

  // Фильтруем данные, чтобы убрать NaN/null значения
  rsiData = rsiData.filter(item => item.rsi !== null && !isNaN(item.rsi));
  macdData = macdData.filter(item => 
    item.macd !== null && !isNaN(item.macd) && 
    item.signal !== null && !isNaN(item.signal)
  );

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
          {rsiData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rsiData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="timestamp" 
                  angle={-45} 
                  textAnchor="end" 
                  height={50}
                  tick={{ fontSize: 12 }}
                />
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
                  activeDot={{ r: 4 }}
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
          ) : (
            <div className="no-data">No RSI data available</div>
          )}
        </div>

        {/* MACD Indicator */}
        <div className="indicator-chart">
          <h4>MACD</h4>
          <p className="indicator-description">
            Moving Average Convergence Divergence - trend-following momentum indicator.
          </p>
          {macdData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={macdData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="timestamp" 
                  angle={-45} 
                  textAnchor="end" 
                  height={50}
                  tick={{ fontSize: 12 }}
                />
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
                  activeDot={{ r: 4 }}
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
          ) : (
            <div className="no-data">No MACD data available</div>
          )}
        </div>
      </div>

      {/* Indicator Summary */}
      <div className="indicator-summary">
        <h4>Current Indicator Values:</h4>
        <div className="indicator-values">
          <div className="indicator-value">
            <span>RSI:</span>
            <strong>
              {hasFullData ? 
                (data.indicators.rsi_14[data.indicators.rsi_14.length - 1]?.toFixed(2) || 'N/A') :
                (data.rsi[data.rsi.length - 1]?.toFixed(2) || 'N/A')
              }
            </strong>
          </div>
          <div className="indicator-value">
            <span>MACD:</span>
            <strong>
              {hasFullData ?
                (data.indicators.macd[data.indicators.macd.length - 1]?.toFixed(4) || 'N/A') :
                (data.macd[data.macd.length - 1]?.toFixed(4) || 'N/A')
              }
            </strong>
          </div>
          <div className="indicator-value">
            <span>MACD Signal:</span>
            <strong>
              {hasFullData ?
                (data.indicators.macd_signal[data.indicators.macd_signal.length - 1]?.toFixed(4) || 'N/A') :
                (data.macd_signal[data.macd_signal.length - 1]?.toFixed(4) || 'N/A')
              }
            </strong>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TechnicalIndicators;