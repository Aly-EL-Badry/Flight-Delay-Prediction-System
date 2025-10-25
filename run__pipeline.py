from pipelines.flight_pipeline import flight_delay_pipeline

if __name__ == "__main__":
    flights_path = "data/flights.csv"
    airlines_path = "data/airlines.csv"
    airports_path = "data/airports.csv"

    pipeline_instance = flight_delay_pipeline(
        flights_path=flights_path,
        airlines_path=airlines_path,
        airports_path=airports_path,
    )

    # Run pipeline
    pipeline_instance.run()
