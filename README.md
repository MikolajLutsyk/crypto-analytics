<div align="center">

# 📊 Crypto Price Prediction Platform

**A comprehensive platform for cryptocurrency price prediction and monitoring using machine learning techniques.**  
The system fetches real-time data from Binance API, performs technical analysis, and generates price movement predictions using the CatBoost algorithm.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-TimescaleDB-336791?style=flat-square&logo=postgresql&logoColor=white)](https://www.timescale.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 🎯 Features

| Feature | Description |
|---|---|
| **Real-time Data Collection** | Automated fetching of OHLCV data from Binance API |
| **Technical Analysis** | Generation of 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.) |
| **ML Predictions** | Binary classification of price direction using CatBoost algorithm |
| **Interactive Dashboard** | React-based UI with real-time charts and metrics |
| **REST API** | FastAPI backend with comprehensive endpoints |
| **Time-series Database** | Optimized storage with PostgreSQL + TimescaleDB |

---

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Binance   │ --> │ Data         │ --> │ TimescaleDB │
│   API       │     │ Collector    │     │             │
└─────────────┘     │ (Python)     │     └─────────────┘
                    └──────────────┘            │
┌─────────────┐     ┌──────────────┐     ┌─────┴─────┐
│   React     │ <-- │   FastAPI    │ <-- │    ML     │
│   Frontend  │     │   Server     │     │   Model   │
└─────────────┘     └──────────────┘     └───────────┘
```

---

## 🛠️ Technology Stack

### Backend
- **Python 3.9+** — Core programming language
- **FastAPI** — REST API framework
- **SQLAlchemy** — Database ORM
- **Pandas / NumPy** — Data processing
- **CatBoost** — ML prediction algorithm
- **TA-Lib** — Technical indicators calculation
- **Joblib** — Model serialization

### Database
- **PostgreSQL** — Relational database
- **TimescaleDB** — Time-series optimization

### Frontend
- **React 18** — UI library
- **Recharts** — Data visualization
- **Axios** — API communication
- **CSS3** — Styling

### Development & Testing
- **pytest** — Unit and integration testing
- **Docker** — Containerization
- **Jupyter Notebook** — Exploratory analysis

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- Node.js 16+
- Docker and Docker Compose
- PostgreSQL *(optional, if not using Docker)*

### Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/crypto-prediction-platform.git
cd crypto-prediction-platform
```

**2. Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

**3. Start the database**
```bash
docker-compose up -d
```

**4. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**5. Run data collector**
```bash
python collector.py
```

**6. Generate features and train model**
```bash
python feature_engineering.py
python train_model.py
```

**7. Start API server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**8. Install frontend dependencies and start**
```bash
cd frontend
npm install
npm start
```

> 🌐 The application will be available at **http://localhost:3000**

---

## 📊 Data Pipeline

### 1. Data Collection
- Fetches hourly OHLCV data for BTC/USDT from Binance
- Automatic pagination handling (1000 records per request)
- Deduplication mechanism to avoid duplicates

### 2. Feature Engineering

| Category | Indicators |
|---|---|
| **Trend** | SMA, EMA, MACD |
| **Momentum** | RSI (7, 14, 21 periods) |
| **Volatility** | ATR, Bollinger Bands |
| **Volume** | OBV, VWAP |
| **Time** | Hour, day of week (sin/cos encoding) |
| **Lag** | 1–48 hour lags of key variables |

### 3. Model Training
- Binary classification (price up/down in next hour)
- Time-based train/test split (80/20)
- CatBoost with early stopping and class balancing
- Feature selection using ANOVA F-test

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/ohlc` | `GET` | Get OHLCV data with technical indicators |
| `/api/prediction` | `GET` | Get latest price prediction |
| `/api/metrics` | `GET` | Get model performance metrics |
| `/api/feature-importance` | `GET` | Get feature importance rankings |
| `/health` | `GET` | Health check |

---

## 📈 Model Performance

> CatBoost model metrics evaluated on held-out test data:

| Metric | Value |
|---|---|
| **Accuracy** | 63.3% |
| **Balanced Accuracy** | 63.5% |
| **Precision** | 63.7% |
| **Recall** | 63.3% |
| **F1-Score** | 63.4% |

---

## 🧪 Testing

Run the full test suite:
```bash
pytest tests/ -v
```

**Test categories:**
- API endpoint availability
- Input validation
- Response structure
- Error handling
- Data integrity

---

## 🐳 Docker Support

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Containers
```bash
# Database only
docker run -d \
  --name crypto-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=crypto \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg14
```

---

## ⚙️ Configuration

Key configuration parameters in `config.py`:

```python
# Database
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "crypto"

# Data collection
SYMBOL       = "BTCUSDT"
INTERVAL     = "1h"
LOOKBACK_DAYS = 180

# Model
N_FEATURES   = 20
TEST_SIZE    = 0.2
RANDOM_STATE = 42
```

---

## 📁 Project Structure

```
crypto-prediction-platform/
├── backend/
│   ├── collector.py           # Data collection module
│   ├── feature_engineering.py # Feature generation
│   ├── train_model.py         # Model training
│   ├── main.py                # FastAPI server
│   └── models/                # Saved models
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── services/          # API services
│   │   └── App.js             # Main component
│   └── package.json
├── tests/
│   └── test_api.py            # API tests
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔍 Key Implementation Details

### Data Deduplication
```python
last_timestamp = pd.read_sql("SELECT MAX(open_time) FROM ohlcv", engine).iloc[0, 0]
new_data = df[df['open_time'] > last_timestamp]
```

### Feature Selection
```python
selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
X_selected = selector.fit_transform(X, y)
```

### Model Training with Early Stopping
```python
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    depth=6,
    auto_class_weights='Balanced',
    od_type='Iter',
    od_wait=50
)
```

---

## 🚀 Performance Optimization

- **Database** — TimescaleDB hypertables for time-series data
- **API** — FastAPI async endpoints
- **Frontend** — React component memoization
- **ML** — Model caching in memory

---

## 🔒 Error Handling

The system implements comprehensive error handling:

- Exponential backoff for API rate limits
- Transaction rollbacks for database operations
- Fallback strategies for component failures
- Centralized logging with error codes

---

## 📚 Future Enhancements

- [ ] Multi-cryptocurrency support
- [ ] LSTM / GRU deep learning models
- [ ] Sentiment analysis from social media
- [ ] Backtesting module for strategy validation
- [ ] User authentication and personalized dashboards
- [ ] Mobile app with push notifications
- [ ] Ensemble learning with multiple algorithms

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Mykola Lutsyk**  
Engineering Thesis, Computer Science — Software Engineering specialization  
Supervisor: Dr. Paweł Powróźnik  
Lublin, 2026

---

## 🙏 Acknowledgments

- [Binance](https://binance.com) for providing free API access
- [TimescaleDB](https://timescale.com) for time-series optimization
- [CatBoost](https://catboost.ai) developers for the excellent ML library
- All open-source libraries used in this project
