import React, { useState, useEffect } from 'react';
import { cryptoAPI } from './services/api';
import PriceChart from './components/PriceChart';
import FeatureImportance from './components/FeatureImportance';
import ModelMetrics from './components/ModelMetrics';
import TechnicalIndicators from './components/TechnicalIndicators';
import './App.css';

function App() {
  const [chartData, setChartData] = useState(null);
  const [features, setFeatures] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [symbol, setSymbol] = useState('BTCUSDT');

  const loadData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const [chartResponse, featuresResponse, metricsResponse, predictionResponse] = await Promise.all([
        cryptoAPI.getOHLCV(symbol),
        cryptoAPI.getFeatureImportance(20),
        cryptoAPI.getModelMetrics(),
        cryptoAPI.getCurrentPrediction()
      ]);

      setChartData(chartResponse.data);
      setFeatures(featuresResponse.data.features);
      setMetrics(metricsResponse.data);
      setPrediction(predictionResponse.data);
    } catch (err) {
      console.error('Error loading data:', err);
      setError('Failed to load data. Make sure the backend server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [symbol]);

  const handleRefresh = () => {
    loadData();
  };

  if (loading) {
    return (
      <div className="App">
        <div className="loading-full">
          <div className="spinner"></div>
          <h2>Loading Crypto Analytics Dashboard...</h2>
          <p>This may take a few moments</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="App">
        <div className="error-container">
          <h2>Error</h2>
          <p>{error}</p>
          <button onClick={handleRefresh}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>Crypto Analytics Dashboard</h1>
          <p>Real-time cryptocurrency analysis with machine learning predictions</p>
        </div>
        <div className="controls">
          <div className="control-group">
            <label htmlFor="symbol-select">Select Symbol:</label>
            <select 
              id="symbol-select"
              value={symbol} 
              onChange={(e) => setSymbol(e.target.value)}
            >
              <option value="BTCUSDT">BTC/USDT</option>
              <option value="ETHUSDT">ETH/USDT</option>
            </select>
          </div>
          <button onClick={handleRefresh} className="refresh-btn">
            Refresh Data
          </button>
        </div>
      </header>

      <main className="dashboard">
        {/* Price Chart Section */}
        <section className="dashboard-section">
          <PriceChart data={chartData} />
        </section>

        {/* Technical Indicators Section */}
        <section className="dashboard-section">
          <TechnicalIndicators data={chartData} />
        </section>

        {/* ML Model Section */}
        <section className="dashboard-section">
          <div className="row">
            <div className="col-half">
              <FeatureImportance features={features} />
            </div>
            <div className="col-half">
              <ModelMetrics metrics={metrics} prediction={prediction} />
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Crypto Analytics Dashboard | 
          Data from Binance API | 
          ML Model: CatBoost | 
          Built with FastAPI & React
        </p>
      </footer>
    </div>
  );
}

export default App;