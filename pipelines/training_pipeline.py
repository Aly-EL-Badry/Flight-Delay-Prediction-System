from zenml import pipeline
from steps.clean_data import clean_df
from steps.ingest_data import ingest_flights_data, ingest_airlines_data, ingest_airports_data

@pipeline(enable_cache=False)
def train_pipeline(
    flights_path:str,
    airlines_path:str,
    airports_path:str
):
    flights_df = ingest_flights_data(flights_path)
    airlines_df = ingest_airlines_data(airlines_path)
    airports_df = ingest_airports_data(airports_path)  
    
    processed_flights = clean_df(flights_df=flights_df, airports_df=airports_df)
    return processed_flights