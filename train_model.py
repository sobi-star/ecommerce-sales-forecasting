import os,pickle
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def features(x):
    x=x.copy()
    x["month_num"]=x.Date.dt.month
    x["year"]=x.Date.dt.year
    x["quarter"]=x.Date.dt.quarter
    x["time_index"]=np.arange(len(x))
    for lag in [1,2,3,6,12]: x[f"lag_{lag}"]=x.Revenue.shift(lag)
    for w in [3,6,12]: x[f"rolling_{w}"]=x.Revenue.shift(1).rolling(w).mean()
    return x

df=pd.read_csv("data/ecommerce_sales.csv",parse_dates=["Order_Date"])
monthly=df.groupby(pd.Grouper(key="Order_Date",freq="MS")).agg(
 Revenue=("Revenue","sum"),Quantity=("Quantity","sum"),Orders=("Product","count")
).reset_index().rename(columns={"Order_Date":"Date"})
f=features(monthly).dropna().reset_index(drop=True)
cols=["month_num","year","quarter","time_index","lag_1","lag_2","lag_3","lag_6","lag_12","rolling_3","rolling_6","rolling_12"]
split=int(len(f)*.8); tr=f.iloc[:split]; te=f.iloc[split:]
model=RandomForestRegressor(n_estimators=500,max_depth=10,min_samples_leaf=2,random_state=42,n_jobs=-1)
model.fit(tr[cols],tr.Revenue)
pred=model.predict(te[cols])
metrics={"MAE":float(mean_absolute_error(te.Revenue,pred)),
"RMSE":float(np.sqrt(mean_squared_error(te.Revenue,pred))),"R2":float(r2_score(te.Revenue,pred))}
test=te[["Date","Revenue"]].copy();test["Predicted_Revenue"]=pred
test.to_csv("outputs/test_predictions.csv",index=False)

# Recursive six-month forecast
hist=monthly[["Date","Revenue","Quantity","Orders"]].copy(); future=[]
for _ in range(6):
    nd=hist.Date.iloc[-1]+pd.offsets.MonthBegin(1)
    temp=pd.concat([hist,pd.DataFrame([{"Date":nd,"Revenue":np.nan,"Quantity":0,"Orders":0}])],ignore_index=True)
    z=features(temp).iloc[-1].copy()
    y=float(model.predict(pd.DataFrame([z])[cols])[0])
    hist=pd.concat([hist,pd.DataFrame([{"Date":nd,"Revenue":y,"Quantity":0,"Orders":0}])],ignore_index=True)
    future.append([nd,y])
pd.DataFrame(future,columns=["Date","Forecast_Revenue"]).to_csv("outputs/future_forecast.csv",index=False)
with open("outputs/metrics.txt","w") as out:
    for k,v in metrics.items(): out.write(f"{k}: {v:.4f}\n")
with open("models/sales_model.pkl","wb") as out:
    pickle.dump({"model":model,"features":cols,"monthly":monthly,"metrics":metrics},out)
print(metrics)
