import ssl
import json
import pandas as pd
from prophet import Prophet
from datetime import date, datetime
import holidays
import pandas as pd
import requests
# Graphs leh Charts plot nan
import matplotlib.pyplot as pp
from matplotlib import pyplot
from prophet.plot import plot_plotly, plot_components_plotly
import pymongo
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px



url = 'https://api.jsonbin.io/v3/b'
headers = {
  'Content-Type': 'application/json',
  'X-Master-Key': '$2b$10$a/cr4tTLVkQHKYf9yjB95uiWMG2b.A6xTa3SV/uHeQRh9.FEotO/e'
  }

client = pymongo.MongoClient("mongodb://localhost:27017")


df = pd.read_csv('water_demand_dataset.csv')

data = df.to_dict(orient = "records")
db = client["Machinelearning"]
print(db)
db.water_demand_dataset.insert_many(data)
# Holiday list kan siam ang!
year="2023"
# Ram Hming lak tur
holidays_india=holidays.India()
# Add Holiday without description
start = year + "-01-01"
periods = 52
freq = "W-SUN"
  
# Sunday zawng zawng
sundays = pd.date_range(start = start,periods = periods,freq = freq)

"""for aa in sundays.strftime('%d-%m-%Y'):
    print(aa)
    holidays_india_dataframe=pd.DataFrame({
    "holiday":"Sunday",
    "ds":aa})"""


m = Prophet()

""" Kan rama Holidays te a lo telh theih vek a nih chu!!!... mahse mawww.... Diwali leh holi chiah an telh niaa.. chu pawh  from 2010 to 2030!!!!...whmmmmppppps"""


#m.add_country_holidays(country_name='IN')

""" Hemi Block hi kan dataset a felfai chuan a ngai lo """
df2=df.copy()
print(df2.columns)

df2['ds'] = pd.to_datetime(df2['ds'], format="%d-%m-%Y %H:%M")


""" zing dar 7 hmalam data chu kan duh lo"""
df2 = df2[df2['ds'].dt.hour > 7 ]

df3=df2.copy()
df3['ds'] = pd.to_datetime(df3['ds'])

""" Tlai dar 6 hmalam data chu kan duh lo"""
df3 = df3[df3['ds'].dt.hour < 19 ]
""" Block Tawpna """

df3 = df3[df3['ds'].dt.dayofweek < 6 ] # Sunday data hi kan training data atan kan hmang tel lo ang!
""" Kan dataframe kha Prophet-ah kan fit anf= kan barh ang"""

print(df3.describe())
m.fit(df3)

""" Hei hi a dik reng """

future = m.make_future_dataframe(periods=120,freq="H")

future2 = future.copy()

""" KAn Future data tur hian zing dar 7 hmalam data chu kan duh lo"""
future2 = future2[future2['ds'].dt.hour > 7]
future3=future2.copy()

""" KAn Future data tur hian Tlai dar 6 hnulam data chu kan duh loooooooo"""
future3 = future3[future3['ds'].dt.hour < 19]

future3=future3[future3['ds'].dt.dayofweek < 6]# Sunday kan predict tel lo ang!!

""" Kan Future data(frame) remdik hmangin kan predict ang!!!!!"""

forecast = m.predict(future3)

""" Kan Forecast data hi Json file-ah kan save ang"""
""" a Key kan dah luh tur kan filter ang... array?(python hian array index-ah integer ni lo hman a phal ve tlatsss... ha haaa) indexing...."""


#jsondata=forecast[["yhat","yhat_lower"]].tail(10).to_json(orient='columns') 

#print(jsondata)
#jsondata1={jsondata}
jsondata=forecast["yhat"].tail(10).to_json(orient='index')

#jsondata={"878":5.3702528002,"879":5.2568040727,"880":5.4392679036,"881":5.5571748762,"882":5.4937496183,"883":5.4185879442,"884":5.5087732552,"885":5.6999196391,"886":5.751550795,"887":5.5417643562}

""" Json file kan siam ang!!!.. WRITE-access nei turin!!!!!! """

with open("forecastjson.json", "w") as outfile:
    outfile.write(jsondata)

""" Forecast Data En ve chhin hrim hrim ang!!!"""

#f = open('forecastjson.json')

#datajson=json.load(f)
#req=requests.post(url, json={"333":5.3702528002,"879":5.2568040727,"880":5.4392679036,"881":5.5571748762,"882":5.4937496183,"883":5.4185879442,"884":5.5087732552,"885":5.6999196391,"886":5.751550795,"887":5.5417643562} , headers=headers)
#print(req.text)

print(forecast.tail(30))


 
pp.plot(df3['y'],label="Actual Data")
pp.plot(forecast['yhat'],label="Predicted")
pp.legend()
fig1 = m.plot(forecast,xlabel="Prediction",include_legend=True)

#fig2 = m.plot_components(forecast)

#plot_plotly(m, forecast)

#plot_components_plotly(m, forecast) 
pyplot.legend()

""" Kan Graph a lan theih nan!!!!... Anaconda Spyder ka hmanin hei hi a ngai bawk si lo aaaa..min ti buai latukkkkk!!!! """
pyplot.show()

