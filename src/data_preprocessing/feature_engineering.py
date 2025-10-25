import pandas as pd
import logging
def time_to_minutes(x):
    if pd.isnull(x):
        return None
    x = int(x)
    hours = x // 100
    minutes = x % 100
    return hours * 60 + minutes

def feature_engineering(df:pd.DataFrame,airportData:pd.DataFrame)->pd.DataFrame:
   #Encode airlines col
   df = pd.get_dummies(df, columns=['AIRLINE'], drop_first=True) 
   

   # Clean airport data (convert categorical)
   airportData['STATE'] = airportData['STATE'].astype('category').cat.codes
   airportData['CITY'] = airportData['CITY'].astype('category').cat.codes 
   

   # drop numeric origin airports
   df = df[~df['ORIGIN_AIRPORT'].astype(str).str.isnumeric()]

   # mapping city , state and lang and lat to the main dataframe for the origin airport and destination airport
   df = df.merge(airportData[['IATA_CODE', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']], left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='left')
   df = df.merge(airportData[['IATA_CODE', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']], left_on='DESTINATION_AIRPORT', right_on='IATA_CODE', how='left', suffixes=('_origin', '_destination'))

   #Drop merged code columns
   df.drop(columns=['IATA_CODE_origin', 'IATA_CODE_destination'], inplace=True, axis=1)
   

   # Encode airport categorical IDs
   df["ORIGIN_AIRPORT"] = df["ORIGIN_AIRPORT"].astype('category').cat.codes
   df["DESTINATION_AIRPORT"] = df["DESTINATION_AIRPORT"].astype('category').cat.codes
   
   # Convert times to minutes
   time_columns = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL']
   for col in time_columns:
        df[col] = df[col].apply(time_to_minutes)

   df.to_csv('../../data/processed/flights_final_processed.csv', index=False)
   logging.info("Flights feature-engineered data saved to '../../data/processed/flights_final_processed.csv'")

   logging.info("Feature engineering completed successfully.")
   return df



