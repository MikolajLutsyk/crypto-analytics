import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

const FeatureImportance = ({ features }) => {
  if (!features || features.length === 0) {
    return <div className="loading">Loading feature importance data...</div>;
  }

  // Сортируем фичи по важности
  const sortedFeatures = [...features].sort((a, b) => b.importance - a.importance);

  const formatTooltip = (value) => {
    return [value.toFixed(4), 'Importance'];
  };

  return (
    <div className="chart-container">
      <h3>🎯 Feature Importance (CatBoost Model)</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={sortedFeatures}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 150, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            type="number" 
            domain={[0, 'dataMax']}
            tickFormatter={(value) => value.toFixed(3)}
          />
          <YAxis 
            type="category" 
            dataKey="feature" 
            width={140}
            tick={{ fontSize: 12 }}
          />
          <Tooltip formatter={formatTooltip} />
          <Bar 
            dataKey="importance" 
            fill="#8884d8" 
            name="Importance"
            radius={[0, 4, 4, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
      
      <div className="features-summary">
        <h4>Top 5 Most Important Features:</h4>
        <ol>
          {sortedFeatures.slice(0, 5).map((feature, index) => (
            <li key={feature.feature}>
              <strong>{feature.feature}</strong>: {feature.importance.toFixed(4)}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
};

export default FeatureImportance;