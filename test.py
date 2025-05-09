import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# For engine size model
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)

y_test_ = regressor.predict(X_test.reshape(-1,1))
print("Engine Size Model - R2-score: %.2f" % r2_score(y_test_, y_test))

# For fuel consumption model
X1 = cdf.FUELCONSUMPTION_COMB.to_numpy()
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.2, random_state=42)
regressor01 = linear_model.LinearRegression()
regressor01.fit(X1_train.reshape(-1, 1), y1_train)

y1_test_ = regressor01.predict(X1_test.reshape(-1,1))
print("Fuel Consumption Model - R2-score: %.2f" % r2_score(y1_test_, y1_test))

# For cylinders model
X2 = cdf.CYLINDERS.to_numpy()
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)
regressor02 = linear_model.LinearRegression()
regressor02.fit(X2_train.reshape(-1, 1), y2_train)

y2_test_ = regressor02.predict(X2_test.reshape(-1,1))
print("Cylinders Model - R2-score: %.2f" % r2_score(y2_test_, y2_test))

# For a fourth model (multiple regression with both engine size and fuel consumption)
X3 = np.column_stack((cdf.ENGINESIZE.to_numpy(), cdf.FUELCONSUMPTION_COMB.to_numpy()))
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.2, random_state=42)
regressor03 = linear_model.LinearRegression()
regressor03.fit(X3_train, y3_train)

y3_test_ = regressor03.predict(X3_test)
print("Multiple Regression Model - R2-score: %.2f" % r2_score(y3_test_, y3_test))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# First subplot for Engine Size
ax1.scatter(X_train, y_train, color='blue', label='Training Data')
ax1.scatter(X_test, y_test, color='green', label='Test Data')
X_sorted = np.sort(np.concatenate((X_train, X_test)))
y_pred_sorted = regressor.predict(X_sorted.reshape(-1, 1))
ax1.plot(X_sorted, y_pred_sorted, '-r')
ax1.set_xlabel("Engine Size")
ax1.set_ylabel("CO2 Emission")
ax1.set_title("Engine Size vs. CO2 Emission")
ax1.legend()

# Second subplot for Fuel Consumption
ax2.scatter(X1_train, y1_train, color='blue', label='Training Data')
ax2.scatter(X1_test, y1_test, color='green', label='Test Data')
X1_sorted = np.sort(np.concatenate((X1_train, X1_test)))
y1_pred_sorted = regressor01.predict(X1_sorted.reshape(-1, 1))
ax2.plot(X1_sorted, y1_pred_sorted, '-r')
ax2.set_xlabel("Fuel Consumption")
ax2.set_ylabel("CO2 Emission")
ax2.set_title("Fuel Consumption vs. CO2 Emission")
ax2.legend()

# Third subplot for Cylinders
ax3.scatter(X2_train, y2_train, color='blue', label='Training Data')
ax3.scatter(X2_test, y2_test, color='green', label='Test Data')
X2_sorted = np.sort(np.concatenate((X2_train, X2_test)))
y2_pred_sorted = regressor02.predict(X2_sorted.reshape(-1, 1))
ax3.plot(X2_sorted, y2_pred_sorted, '-r')
ax3.set_xlabel("Cylinders")
ax3.set_ylabel("CO2 Emission")
ax3.set_title("Cylinders vs. CO2 Emission")
ax3.legend()

# Fourth subplot for Multiple Regression (visualized using predicted vs actual)
ax4.scatter(y3_train, regressor03.predict(X3_train), color='blue', label='Training Data')
ax4.scatter(y3_test, y3_test_, color='green', label='Test Data')
ax4.plot([min(y), max(y)], [min(y), max(y)], 'r-', lw=2) # Diagonal line for perfect predictions
ax4.set_xlabel("Actual CO2 Emission")
ax4.set_ylabel("Predicted CO2 Emission")
ax4.set_title("Multiple Regression: Actual vs Predicted")
ax4.legend()

fig.suptitle("Comparison of Regression Models", fontsize=16)

plt.tight_layout()
plt.show()