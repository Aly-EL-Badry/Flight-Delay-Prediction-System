import pandas as pd
import logging
import os
def clean_flights_data(df:pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary cols
    Drop_cols={'CANCELLATION_REASON', "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY","FLIGHT_NUMBER", "TAIL_NUMBER", "YEAR","CANCELLED","DIVERTED"}
    df.drop(columns=Drop_cols, inplace=True, axis=1, errors='ignore')

    # Some Targets that we are not interested in 
    targets = ["ELAPSED_TIME", "AIR_TIME", "WHEELS_ON","WHEELS_OFF","TAXI_IN","TAXI_OUT","ARRIVAL_TIME"]
    df.drop(columns=targets, inplace=True, axis=1, errors='ignore')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop null
    df.dropna(inplace=True)
    

    # Save preprocessed (cleaned) data before feature engineering
    os.makedirs('../../data/processed', exist_ok=True)
    df.to_csv('../../data/processed/flights_preprocessed.csv', index=False)
    logging.info(" Flights preprocessed data saved to '../../data/processed/flights_preprocessed.csv'")

    logging.info("Flights data cleaning completed successfully.")
    return df




   