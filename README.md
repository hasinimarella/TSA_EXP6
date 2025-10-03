# TSA_EXP6
# Ex.No: 6 HOLT WINTERS METHOD
## Date:30-09-2025
## AIM:
## ALGORITHM:
1.You import the necessary libraries
2.You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, and perform some initial data exploration
3.You group the data by date and resample it to a monthly frequency (beginning of the month
4.You plot the time series data
5.You import the necessary 'statsmodels' libraries for time series analysis
6.You decompose the time series data into its additive components and plot them:
7.You calculate the root mean squared error (RMSE) to evaluate the model's performance
8.You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt- Winters model to the entire dataset and make future predictions
9.You plot the original sales data and the predictions
## PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
data = pd.read_csv('/content/blood_donor_dataset.csv', parse_dates=['created_at'], index_col='created_at') 
data.head()
data_monthly = data.resample('MS')['pints_donated'].sum() #Month start - selecting only 'pints_donated'
data_monthly.head()
data_monthly.plot()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), index=data_monthly.index) # Added missing parenthesis and index assignment
scaled_data = scaled_data + 1e-9 # Add small constant for multiplicative seasonality
scaled_data.plot() # The data seems to have additive trend and multiplicative seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
plt.show()
np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit() # Fixed syntax and added seasonal_periods
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4)) #for next year
ax=data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Months') # Corrected xlabel
ax.set_ylabel('Number of monthly passengers') # Corrected ylabel
ax.set_title('Prediction')
plt.show()
```
## OUTPUT:
Scaled_data plot:
<img width="710" height="529" alt="Screenshot 2025-10-03 111154" src="https://github.com/user-attachments/assets/bb6a597a-e0be-413d-8550-7ae3331ee2e7" />
Decomposed plot:
<img width="857" height="592" alt="image" src="https://github.com/user-attachments/assets/e95d4ec4-a132-4267-b2c4-709f12ff90e8" />
TEST_PREDICTION:
<img width="712" height="570" alt="image" src="https://github.com/user-attachments/assets/6272b4d9-d3f5-4252-9d12-9dad252aecd0" />

Model performance metrics:

Rmse:
<img width="261" height="20" alt="image" src="https://github.com/user-attachments/assets/aa77baed-b19d-4064-b3d8-88cd43ff7e9b" />

Standard deviation and mean:
<img width="683" height="20" alt="image" src="https://github.com/user-attachments/assets/e59bcb25-39dd-440c-bf46-fd02e92240b8" />

FINAL_PREDICTION:

<img width="811" height="571" alt="image" src="https://github.com/user-attachments/assets/a1084103-9196-4817-8270-78e8abc925ce" />


## RESULT:
Thus the program run successfully based on the Holt Winters Method model
