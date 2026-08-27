import os,pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="E-Commerce Sales Intelligence",page_icon="📈",layout="wide")
st.title("📈 E-Commerce Sales Intelligence Dashboard")
st.markdown("### Machine Learning powered revenue forecasting & business analytics")
st.caption("Historical transaction analysis • Random Forest Regression • 6-month forecast")

with open("models/sales_model.pkl","rb") as f: bundle=pickle.load(f)
df=pd.read_csv("data/ecommerce_sales.csv",parse_dates=["Order_Date"])
forecast=pd.read_csv("outputs/future_forecast.csv",parse_dates=["Date"])
test=pd.read_csv("outputs/test_predictions.csv",parse_dates=["Date"])
model=bundle["model"]; metrics=bundle["metrics"]; features=bundle["features"]

st.sidebar.header("🔎 Dashboard Filters")
lo,hi=df.Order_Date.min().date(),df.Order_Date.max().date()
dr=st.sidebar.date_input("Order date range",(lo,hi),min_value=lo,max_value=hi)
cats=sorted(df.Category.unique())
sel=st.sidebar.multiselect("Categories",cats,default=cats)
if isinstance(dr,tuple) and len(dr)==2: a,b=pd.Timestamp(dr[0]),pd.Timestamp(dr[1])
else: a,b=pd.Timestamp(lo),pd.Timestamp(hi)
x=df[(df.Order_Date>=a)&(df.Order_Date<=b)&df.Category.isin(sel)]
rev=x.Revenue.sum(); units=x.Quantity.sum(); orders=len(x); aov=rev/orders if orders else 0
m=x.groupby(pd.Grouper(key="Order_Date",freq="MS")).Revenue.sum()
growth=((m.iloc[-1]-m.iloc[-2])/m.iloc[-2]*100) if len(m)>1 and m.iloc[-2] else 0
k=st.columns(5)
k[0].metric("💰 Total Revenue",f"Rs. {rev:,.0f}");k[1].metric("📦 Units Sold",f"{units:,.0f}")
k[2].metric("🛒 Transactions",f"{orders:,}");k[3].metric("💳 Avg. Order Value",f"Rs. {aov:,.0f}")
k[4].metric("📈 Latest MoM Growth",f"{growth:+.1f}%")
st.divider()

t1,t2,t3,t4=st.tabs(["📊 Business Analytics","🔮 Sales Forecast","🤖 Model Performance","ℹ️ Methodology"])
with t1:
    st.subheader("Historical Monthly Revenue")
    monthly=x.groupby(pd.Grouper(key="Order_Date",freq="MS")).Revenue.sum()
    st.line_chart(monthly,height=350)
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Revenue by Category");st.bar_chart(x.groupby("Category").Revenue.sum().sort_values(ascending=False))
    with c2:
        st.subheader("Top 10 Products");st.bar_chart(x.groupby("Product").Revenue.sum().sort_values(ascending=False).head(10))
    st.subheader("Category Summary")
    s=x.groupby("Category").agg(Revenue=("Revenue","sum"),Units=("Quantity","sum"),Transactions=("Product","count"),Avg_Price=("Unit_Price","mean"),Avg_Discount=("Discount","mean")).sort_values("Revenue",ascending=False)
    st.dataframe(s.style.format({"Revenue":"Rs. {:,.0f}","Units":"{:,.0f}","Transactions":"{:,.0f}","Avg_Price":"Rs. {:.2f}","Avg_Discount":"{:.1%}"}),use_container_width=True)

with t2:
    st.subheader("🔮 Next 6 Months Revenue Forecast")
    st.info("Forecasts are generated from historical revenue, lag features, rolling averages and calendar features.")
    c=st.columns(3)
    c[0].metric("6-Month Forecast",f"Rs. {forecast.Forecast_Revenue.sum():,.0f}")
    c[1].metric("Average Monthly Forecast",f"Rs. {forecast.Forecast_Revenue.mean():,.0f}")
    p=forecast.Forecast_Revenue.idxmax();c[2].metric("Peak Month",forecast.loc[p,"Date"].strftime("%B %Y"))
    st.line_chart(forecast.set_index("Date").Forecast_Revenue,height=360)
    f=forecast.copy();f["Growth_vs_Previous"]=f.Forecast_Revenue.pct_change()*100
    st.dataframe(f.style.format({"Forecast_Revenue":"Rs. {:,.0f}","Growth_vs_Previous":"{:+.2f}%"}),use_container_width=True)
    st.download_button("⬇️ Download Forecast CSV",forecast.to_csv(index=False),"six_month_sales_forecast.csv","text/csv")

with t3:
    st.subheader("🤖 Machine Learning Model Performance")
    c=st.columns(3);c[0].metric("MAE",f"Rs. {metrics['MAE']:,.0f}");c[1].metric("RMSE",f"Rs. {metrics['RMSE']:,.0f}");c[2].metric("R² Score",f"{metrics['R2']:.3f}")
    st.caption("Lower MAE/RMSE is better; higher R² is better.")
    st.subheader("Actual vs Predicted Revenue")
    z=test.set_index("Date")[["Revenue","Predicted_Revenue"]];z.columns=["Actual Revenue","Predicted Revenue"];st.line_chart(z,height=350)
    st.subheader("Prediction Error")
    e=test.copy();e["Error"]=e.Revenue-e.Predicted_Revenue;st.bar_chart(e.set_index("Date").Error,height=260)
    st.subheader("Feature Importance")
    imp=pd.DataFrame({"Feature":features,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False).head(12)
    st.bar_chart(imp.set_index("Feature").Importance,height=330)
    st.download_button("⬇️ Download Test Predictions",test.to_csv(index=False),"test_predictions.csv","text/csv")

with t4:
    st.subheader("📚 Project Methodology")
    st.markdown("""### Business Problem
Predict future e-commerce revenue to support inventory, marketing and financial planning.

### Data Preparation
Transaction data is aggregated into monthly revenue.

### Feature Engineering
Month, year, quarter, time index, 1/2/3/6/12-month lag revenue and 3/6/12-month rolling averages.

### Machine Learning
A **Random Forest Regressor** learns historical revenue patterns.

### Evaluation
A chronological 80/20 train-test split avoids future-data leakage. Metrics: MAE, RMSE and R².

### Forecasting
The trained model recursively predicts the next **6 months**.

### Business Applications
Inventory planning • Sales targets • Marketing campaigns • Budget planning • Product strategy

**Dataset note:** The included dataset is synthetic for demonstration/academic use. Replace it with real store data for real-world predictions.
""")
