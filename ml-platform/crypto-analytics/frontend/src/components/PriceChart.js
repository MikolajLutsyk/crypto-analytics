import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Area 
} from 'recharts';

const PriceChart = ({ data }) => {
  if (!data) return <div className="loading">Loading chart data...</div>;

  // Подготавливаем данные для графика
  const chartData = data.timestamps.map((timestamp, index) => ({
    timestamp: new Date(timestamp).toLocaleDateString(),
    time: new Date(timestamp),
    price: data.prices[index],
    sma_7: data.indicators.sma_7[index],
    sma_25: data.indicators.sma_25[index],
    volume: data.volume[index]
  }));

  const formatTooltip = (value, name) => {
    if (name === 'price' || name === 'sma_7' || name === 'sma_25') {
      return [`$${value.toFixed(2)}`, name];
    }
    if (name === 'volume') {
      return [value.toFixed(2), name];
    }
    return [value, name];
  };

  return (
    <div className="chart-container">
      <h3>📈 Price Chart with Moving Averages</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="timestamp" 
            angle={-45}
            textAnchor="end"
            height={60}
            interval="preserveStartEnd"
          />
          <YAxis 
            yAxisId="left"
            domain={['auto', 'auto']}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />
          <Tooltip 
            formatter={formatTooltip}
            labelFormatter={(label) => `Date: ${label}`}
          />
          <Legend />
          
          {/* Основная линия цены */}
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="price" 
            stroke="#8884d8" 
            name="BTC Price" 
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
          
          {/* SMA 7 */}
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="sma_7" 
            stroke="#82ca9d" 
            name="SMA 7" 
            strokeWidth={1.5}
            dot={false}
          />
          
          {/* SMA 25 */}
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="sma_25" 
            stroke="#ffc658" 
            name="SMA 25" 
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;