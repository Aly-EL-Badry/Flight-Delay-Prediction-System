from zenml import pipeline
from ..steps.DataIngestion import data_ingestion_step
from ..steps.DataHandling import DataCleaning, SpiltStep, fitPreprocessingPipeline, applyPreprocessingPipeline
from ..steps.TrainingModel import trainModel
from ..steps.Evalution import evaluateModel

@pipeline()
def flight_delay_training_pipeline(configData):
    # Ingest data
    raw_data = data_ingestion_step()

    # Clean data
    cleaned_data = DataCleaning(data=raw_data, dropColumns=['column_to_drop1', 'column_to_drop2'])

    # Split data
    X_train, X_test, y_train, y_test = SpiltStep(data=cleaned_data, target='weather', test_size=0.2)

    # Fit preprocessing pipeline
    preprocessing_pipeline = fitPreprocessingPipeline()

    # Apply preprocessing pipeline
    X_train_processed = applyPreprocessingPipeline(data=X_train, pipeline=preprocessing_pipeline)
    X_test_processed = applyPreprocessingPipeline(data=X_test, pipeline=preprocessing_pipeline)



