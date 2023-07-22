import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
data = pd.read_csv('historical_data.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' as the index for time series data
data.set_index('Date', inplace=True)

# Ensure the data is sorted chronologically
data = data.sort_index()

# Visualize the data (optional)
import matplotlib.pyplot as plt
plt.plot(data.index, data['Value'])
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Data')
plt.show()



# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define the ARIMA model (p, d, q)
p, d, q = 5, 1, 0
arima_model = ARIMA(train['Value'], order=(p, d, q))

# Fit the ARIMA model
arima_model_fit = arima_model.fit()

# Forecast future values
forecast_values = arima_model_fit.forecast(steps=len(test))

# Visualize the forecasted values
plt.plot(test.index, test['Value'], label='Actual')
plt.plot(test.index, forecast_values, label='ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()

# Prepare the data for Prophet
prophet_data = data.reset_index().rename(columns={'Date': 'ds', 'Value': 'y'})

# Create and fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_data)

# Create a future dataframe for forecasting
future = prophet_model.make_future_dataframe(periods=len(test))

# Forecast future values with Prophet
forecast = prophet_model.predict(future)

# Visualize the forecasted values
fig = prophet_model.plot(forecast)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Prophet Model Forecast')
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Forecast future values with Prophet
future = prophet_model.make_future_dataframe(periods=len(test))
forecast = prophet_model.predict(future)

# Extract the forecasted values for the test set
forecast_test = forecast[-len(test):]

# Calculate RMSE and MAE for Prophet
prophet_rmse = mean_squared_error(test['Value'], forecast_test['yhat'], squared=False)
prophet_mae = mean_absolute_error(test['Value'], forecast_test['yhat'])

print(f'Prophet RMSE: {prophet_rmse}')
print(f'Prophet MAE: {prophet_mae}')

# Calculate RMSE and MAE for ARIMA
arima_rmse = mean_squared_error(test['Value'], forecast_values, squared=False)
arima_mae = mean_absolute_error(test['Value'], forecast_values)

# Calculate RMSE and MAE for Prophet
prophet_rmse = mean_squared_error(test['Value'], forecast['yhat'], squared=False)
prophet_mae = mean_absolute_error(test['Value'], forecast['yhat'])

print(f'ARIMA RMSE: {arima_rmse}')
print(f'ARIMA MAE: {arima_mae}')
print(f'Prophet RMSE: {prophet_rmse}')
print(f'Prophet MAE: {prophet_mae}')
