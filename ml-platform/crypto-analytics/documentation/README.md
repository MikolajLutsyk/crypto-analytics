для запуска контейнера базы:


для подключения к базе через строку:

psql -h localhost -U postgres -d crypto



crypto-analytics/
├── 📊 BACKEND & DATA
│   ├── collector.py                 # ваш файл - сбор данных с Binance
│   ├── feature_engineering.py       # ваш файл - ETL и фичи
│   ├── train_model.py              # ваш файл - обучение CatBoost
│   ├── config.py                   # НУЖНО СОЗДАТЬ - настройки
│   ├── improved_catboost_model.pkl  # СОЗДАСТСЯ - обученная модель
│   └── requirements.txt            # НУЖНО СОЗДАТЬ - зависимости
│
├── 🌐 FASTAPI BACKEND
│   ├── main.py                     # НУЖНО СОЗДАТЬ - FastAPI приложение
│   ├── database.py                 # НУЖНО СОЗДАТЬ - работа с БД
│   ├── models.py                   # НУЖНО СОЗДАТЬ - Pydantic модели
│   ├── ml_service.py               # НУЖНО СОЗДАТЬ - ML логика
│   └── requirements.txt            # НУЖНО СОЗДАТЬ - зависимости FastAPI
│
├── ⚛️ REACT FRONTEND
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── PriceChart.js       # НУЖНО СОЗДАТЬ
│   │   │   ├── FeatureImportance.js # НУЖНО СОЗДАТЬ
│   │   │   ├── ModelMetrics.js     # НУЖНО СОЗДАТЬ
│   │   │   └── TechnicalIndicators.js # НУЖНО СОЗДАТЬ
│   │   ├── services/
│   │   │   └── api.js              # НУЖНО СОЗДАТЬ
│   │   ├── App.js                  # НУЖНО СОЗДАТЬ
│   │   ├── App.css                 # НУЖНО СОЗДАТЬ
│   │   └── index.js                # НУЖНО СОЗДАТЬ
│   ├── package.json                # НУЖНО СОЗДАТЬ
│   └── package-lock.json           # СОЗДАСТСЯ
│
├── 🗄️ DATABASE & DATA
│   ├── docker-compose.yml          # ваш файл - TimescaleDB
│   ├── init.sql                    # НУЖНО СОЗДАТЬ - инициализация БД
│   ├── data/                       # папка с данными
│   │   └── features.csv            # СОЗДАСТСЯ - фичи для ML
│   └── backups/                    # (опционально) бэкапы
│
└── 📝 ДОКУМЕНТАЦИЯ
    ├── README.md                   # НУЖНО СОЗДАТЬ
    └── run_instructions.md         # НУЖНО СОЗДАТЬ - инструкции









# 🚀 Crypto Analytics Dashboard

Полнофункциональная платформа для анализа криптовалют с машинным обучением и визуализацией в реальном времени.

## 📋 Функциональность

- 📊 **Сбор данных** с Binance API
- ⚙️ **Feature Engineering** - технические индикаторы и фичи
- 🤖 **ML Модель** - CatBoost для предсказания направления цены
- 🌐 **FastAPI Backend** - REST API для данных и предсказаний
- ⚛️ **React Frontend** - интерактивные дашборды и графики
- 🗄️ **TimescaleDB** - хранение временных рядов
