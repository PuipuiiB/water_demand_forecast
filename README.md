Major-Project
The purpose of this project is to deliver a machine learning solution to forecasting aggregate water demand. This work was led by National Informatic centre, Mizoram. This repository contains the code used to fit a machine learning models to predict future daily water consumption. This repository is intended to serve as an example for other municipalities who wish to explore water demand forecasting in their own locales.

Water Demand Forecast Streamlit App
This repository contains code for a Streamlit web application to forecast water demand using the Prophet library. The app fetches a water demand dataset from a CSV file, preprocesses the data, stores it in MongoDB, performs forecasting, and visualizes the results.

Setup
Install the required dependencies:

Ensure that MongoDB is installed and running on your local machine. You can download MongoDB from the official website: MongoDB

Run the Streamlit app:

Description
app.py: This is the main file containing the Streamlit app code. The app fetches the water demand dataset from a CSV file, preprocesses the data, stores it in MongoDB, performs forecasting using the Prophet library, and visualizes the results using interactive plots.
Requirements
Python 3.12
Streamlit
Prophet
Pandas
Matplotlib
Plotly
pymongo
Dataset
The water demand dataset used in this project is stored in a CSV file (water_demand_dataset.csv). It consists of historical water demand data, which is used for forecasting future demand.

Results
The Streamlit app displays descriptive statistics of the dataset, performs forecasting using Prophet, and visualizes the results through interactive plots. Users can interact with the app to explore forecasted demand trends over time.

License
This project is licensed under the MIT License.

Contact
Remruati Bartowski
MCA 6th semester
ph.7005679447

