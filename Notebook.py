import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("advertising.csv")
df.head()
df.shape
df.columns
df.info()
df.isnull().sum()
df.describe()
plt.figure(figsize=(6,4))
sns.scatterplot(x='TV', y='Sales', data=df)
plt.title('TV Advertising vs Sales')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='Radio', y='Sales', data=df)
plt.title('Radio Advertising vs Sales')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='Newspaper', y='Sales', data=df)
plt.title('Newspaper Advertising vs Sales')
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]   # Features
y = df['Sales']                       # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
coeff_df
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mae
mse = mean_squared_error(y_test, y_pred)
mse
r2 = r2_score(y_test, y_pred)
r2

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

new_data = pd.DataFrame({
    'TV': [200],
    'Radio': [25],
    'Newspaper': [30]
})

predicted_sales = model.predict(new_data)
predicted_sales
