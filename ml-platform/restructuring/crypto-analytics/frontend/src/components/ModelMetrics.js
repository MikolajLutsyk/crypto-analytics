import React from 'react';

const ModelMetrics = ({ metrics, prediction }) => {
  if (!metrics) return <div className="loading">Loading model metrics...</div>;

  const metricCards = [
    { key: 'accuracy', label: 'Accuracy', value: metrics.accuracy, color: '#1890ff' },
    { key: 'balanced_accuracy', label: 'Balanced Accuracy', value: metrics.balanced_accuracy, color: '#52c41a' },
    { key: 'precision', label: 'Precision', value: metrics.precision, color: '#faad14' },
    { key: 'recall', label: 'Recall', value: metrics.recall, color: '#f5222d' },
    { key: 'f1_score', label: 'F1-Score', value: metrics.f1_score, color: '#722ed1' }
  ];

  const getPerformanceLevel = (accuracy) => {
    if (accuracy >= 0.7) return { level: 'Excellent', color: '#52c41a' };
    if (accuracy >= 0.6) return { level: 'Good', color: '#1890ff' };
    if (accuracy >= 0.55) return { level: 'Fair', color: '#faad14' };
    return { level: 'Needs Improvement', color: '#f5222d' };
  };

  const performance = getPerformanceLevel(metrics.accuracy);

  return (
    <div className="metrics-container">
      <h3>🤖 Model Performance</h3>
      
      {/* Performance Summary */}
      <div className="performance-summary" style={{ borderLeftColor: performance.color }}>
        <h4>Overall Performance: <span style={{ color: performance.color }}>{performance.level}</span></h4>
        <p>Accuracy: <strong>{(metrics.accuracy * 100).toFixed(1)}%</strong></p>
      </div>

      {/* Metrics Grid */}
      <div className="metrics-grid">
        {metricCards.map(metric => (
          <div key={metric.key} className="metric-card" style={{ borderLeftColor: metric.color }}>
            <h4>{metric.label}</h4>
            <div className="metric-value">{(metric.value * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>

      {/* Current Prediction */}
      {prediction && (
        <div className="prediction-card">
          <h4>🎯 Current Prediction</h4>
          <div className="prediction-details">
            <div className="prediction-item">
              <span>Direction:</span>
              <strong className={prediction.predicted_direction === 'UP' ? 'up' : 'down'}>
                {prediction.predicted_direction}
              </strong>
            </div>
            <div className="prediction-item">
              <span>Confidence:</span>
              <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
            </div>
            <div className="prediction-item">
              <span>Current Price:</span>
              <strong>${prediction.current_price.toFixed(2)}</strong>
            </div>
          </div>
        </div>
      )}

      {/* Confusion Matrix */}
      <div className="confusion-matrix">
        <h4>Confusion Matrix</h4>
        <table>
          <thead>
            <tr>
              <th></th>
              <th>Predicted DOWN</th>
              <th>Predicted UP</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="matrix-label"><strong>Actual DOWN</strong></td>
              <td className="true-negative">{metrics.confusion_matrix[0][0]}</td>
              <td className="false-positive">{metrics.confusion_matrix[0][1]}</td>
            </tr>
            <tr>
              <td className="matrix-label"><strong>Actual UP</strong></td>
              <td className="false-negative">{metrics.confusion_matrix[1][0]}</td>
              <td className="true-positive">{metrics.confusion_matrix[1][1]}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ModelMetrics;