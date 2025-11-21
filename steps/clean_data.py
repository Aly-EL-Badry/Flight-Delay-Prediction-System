import logging
import pandas as pd
from zenml import step
from src.data_preprocessing.clean_flights_data import clean_flights_data
from src.data_preprocessing.feature_engineering import feature_engineering

@step
def clean__df(flights_df,airports_df):
   # Clean raw data
   cleaned_flights=clean_flights_data(flights_df)
   
   # Feature Engineering Mode and outlier adjusting 
   Data = feature_engineering(cleaned_flights,airports_df)

   return Data
