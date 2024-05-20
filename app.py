import ssl
import json
import pandas as pd
from prophet import Prophet
from datetime import date, datetime
import holidays
import requests
import matplotlib.pyplot as pp
from matplotlib import pyplot
from prophet.plot import plot_plotly, plot_components_plotly
import pymongo
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import streamlit as st

# Streamlit App
st.title('Water Demand Forecast')

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017")

# Read dataset
df = pd.read_csv('water_demand_dataset.csv')

# Store data in MongoDB
data = df.to_dict(orient="records")
db = client["Machinelearning"]
db.water_demand_dataset.insert_many(data)

# Set year for holidays
year = "2023"
holidays_india = holidays.India()
start = year + "-01-01"
periods = 52
freq = "W-SUN"
sundays = pd.date_range(start=start, periods=periods, freq=freq)

# Prepare the model
m = Prophet()

# Data preprocessing
df2 = df.copy()
df2['ds'] = pd.to_datetime(df2['ds'], format="%d-%m-%Y %H:%M")
df2 = df2[df2['ds'].dt.hour > 7]
df3 = df2.copy()
df3['ds'] = pd.to_datetime(df3['ds'])
df3 = df3[df3['ds'].dt.hour < 19]
df3 = df3[df3['ds'].dt.dayofweek < 6]

# Display the dataset
st.subheader('Dataset')
st.write(df3.describe())

# Fit the model
m.fit(df3)

# Create future dataframe
future = m.make_future_dataframe(periods=120, freq="H")
future2 = future.copy()
future2 = future2[future2['ds'].dt.hour > 7]
future3 = future2.copy()
future3 = future3[future3['ds'].dt.hour < 19]
future3 = future3[future3['ds'].dt.dayofweek < 6]

# Make predictions
forecast = m.predict(future3)

# Save forecast to JSON file
jsondata = forecast["yhat"].tail(10).to_json(orient='index')
with open("forecastjson.json", "w") as outfile:
    outfile.write(jsondata)

# Display forecast
st.subheader('Forecast')
st.write(forecast.tail(30))

# Plot results
fig1 = m.plot(forecast, xlabel="Prediction", include_legend=True)
st.pyplot(fig1)

# Plotly interactive plot
st.plotly_chart(plot_plotly(m, forecast))

# Plot components
st.plotly_chart(plot_components_plotly(m, forecast))

# Line plot of actual vs predicted
pp.plot(df3['y'], label="Actual Data")
pp.plot(forecast['yhat'], label="Predicted")
pp.legend()
pyplot.legend()
st.pyplot(pp.gcf())
