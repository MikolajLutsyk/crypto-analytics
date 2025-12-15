crypto-analytics/
├── 📊 BACKEND & DATA
│   ├── collector.py                 
│   ├── feature_engineering.py       
│   ├── train_model.py              
│   ├── config.py                   
│   ├── improved_catboost_model.pkl  
│   └── requirements.txt            
│
├── 🌐 FASTAPI BACKEND
│   ├── main.py                     
│   ├── database.py                 
│   ├── models.py                   
│   ├── ml_service.py               
│   └── requirements.txt            
│
├── ⚛️ REACT FRONTEND
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── PriceChart.js       
│   │   │   ├── FeatureImportance.js 
│   │   │   ├── ModelMetrics.js     
│   │   │   └── TechnicalIndicators.js
│   │   ├── services/
│   │   │   └── api.js              
│   │   ├── App.js                 
│   │   ├── App.css                
│   │   └── index.js                
│   ├── package.json               
│   └── package-lock.json           
│
├── 🗄️ DATABASE & DATA
│   ├── docker-compose.yml         
│   ├── init.sql                    
│   ├── data/                      
│   │   └── features.csv            
│   └── backups/                    
│
└── 📝 Documentation
    ├── README.md                  
    └── run_instructions.md         




# 🚀 Crypto Analytics Dashboard

Fully functioning platform for crypto analysis with ML and visualization in real time

📋 Functionality
📊 Data Collection from Binance API
⚙️ Feature Engineering – technical indicators and features
🤖 ML Model – CatBoost for price direction prediction
🌐 FastAPI Backend – REST API for data and predictions
⚛️ React Frontend – interactive dashboards and charts
🗄️ TimescaleDB – time-series data storage