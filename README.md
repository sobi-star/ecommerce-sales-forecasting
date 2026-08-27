# E-Commerce Sales Forecasting — Machine Learning Project

## What this project does
Predicts future monthly e-commerce revenue from historical transactions using a Random Forest regression model.

## Features
- KPI dashboard: revenue, units, transactions, AOV, monthly growth
- Date and category filters
- Historical revenue trend
- Revenue by category
- Top 10 products
- Category summary
- 6-month revenue forecast
- Actual vs predicted chart
- Prediction error analysis
- Feature importance
- CSV downloads
- Methodology/viva explanation

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Dataset
The included CSV is synthetic so the project runs immediately. For a real submission, you can replace it with real store data having the same columns.

## Viva explanation
1. Business problem: forecast future sales.
2. Data: transaction date, category, product, price, discount, quantity and revenue.
3. Feature engineering: lag and rolling-window features plus calendar variables.
4. Model: Random Forest Regressor.
5. Evaluation: chronological 80/20 split, MAE, RMSE and R².
6. Forecast: recursively predict the next six months.
