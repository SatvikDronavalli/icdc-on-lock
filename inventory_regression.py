import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("toyota_camry_sales_400_samples_gas_trend.csv")
X = df.drop(columns=["Predicted_Sales","Month"])
y = df["Predicted_Sales"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_preds = model.predict(X)
'''for i in X.columns:
    plt.scatter(X[i],y,color='blue',edgecolor='k',s=80)
    plt.xlabel(i)
    plt.ylabel("Model Predicted Sales")
    plt.title(f"{i} vs Model Predicted Sales")
    plt.show() '''
print("R² on training data:", r2_score(y,y_preds))

for i in X_test.columns:
    plt.scatter(X_test[i],y_test,color='blue',edgecolor='k',s=80)
    plt.xlabel(i)
    plt.ylabel("Model Predicted Sales")
    plt.title(f"{i} vs Model Predicted Sales")
    plt.show() 
plt.scatter(range(len(y_test)),y_test,color='blue',edgecolor='k',s=80)
plt.scatter(range(len(y_test)),model.predict(X_test),color='red',edgecolor='k',s=80)
plt.show()
r2 = r2_score(y_test, model.predict(X_test))
print("R² on testing data:", r2)
