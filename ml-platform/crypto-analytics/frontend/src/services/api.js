import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

// Создаем экземпляр axios с настройками
const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Интерцептор для обработки ошибок
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const cryptoAPI = {
  getOHLCV: (symbol = 'BTCUSDT', days = 90) => 
    apiClient.get(`/ohlcv/${symbol}?days=${days}`),
  
  getFeatureImportance: (topN = 20) => 
    apiClient.get(`/features/importance?top_n=${topN}`),
  
  getModelMetrics: () => 
    apiClient.get('/model/metrics'),
  
  getCurrentPrediction: () => 
    apiClient.get('/predict/current'),
  
  getTechnicalIndicators: (symbol = 'BTCUSDT', days = 90) => 
    apiClient.get(`/technical/indicators?symbol=${symbol}&days=${days}`)
};