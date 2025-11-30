from zenml import pipeline
from ..steps.DataIngestion import data_ingestion_step
from ..steps.DataHandling import DataCleaning, SpiltStep, fitPreprocessingPipeline, applyPreprocessingPipeline
from ..steps.TrainingModel import trainModel, saveModel, predict
from ..steps.Evalution import evaluateModel, saveMetrics
from ..src.TrainingStrategy.NeuralNetwork.NN import KerasRegressor

@pipeline()
def TrainingPipeline(configData):
    """
    Flight delay training pipeline.

    This pipeline ingests data from a CSV file, cleans the data by dropping specified columns, splits the data into training and test sets, fits a preprocessing pipeline, applies the preprocessing pipeline to the training and test data, and trains a model using the preprocessed data.

    Args:
        configData (dict): Configuration data containing information about the pipeline such as the input data path, preprocessing parameters, and model parameters.

    Returns:
        None
    """
    # Ingest data
    raw_data = data_ingestion_step()

    # Clean data
    DropColsSet = configData["ColumnsToRemove"]
    ColsToDrop = (
        DropColsSet["NullColumns"]
        + DropColsSet["UniqueColumns"]
        + DropColsSet["NoValueColumns"]
        + DropColsSet["Targets"]
        + DropColsSet["TimeColumns"]
    )

    cleaned_data = DataCleaning(data=raw_data, dropColumns=ColsToDrop, ColsToClean= configData["CleaningColumns"])

    # Split data
    X_train, X_test, y_train, y_test = SpiltStep(data=cleaned_data, target=configData["Target"], test_size=0.2)

    # Fit preprocessing pipeline
    preprocessing_pipeline = fitPreprocessingPipeline(data=X_train, 
                                                      LabelCols=configData["FeatureEngCols"]["LabelCols"],
                                                      OheCols  = configData["FeatureEngCols"]["OheCols"],
                                                      ScaleCols= configData["FeatureEngCols"]["ScaleCols"],
                                                      CycleCols= configData["FeatureEngCols"]["CyclicCols"],
                                                      PATH=configData["ProcessingPipelinePath"])

    # Apply preprocessing pipeline
    X_train_processed = applyPreprocessingPipeline(data=X_train, pipeline=preprocessing_pipeline)
    X_test_processed = applyPreprocessingPipeline(data=X_test, pipeline=preprocessing_pipeline)

    trainedModel = trainModel(
        xTrain=X_train_processed,
        yTrain=y_train,
        modelParams=configData["ModelParams"],
        compileParams=configData["compileParams"],
        fitParams=configData["fitParams"]
    )
    
    saveModel(model=trainedModel, savePath=configData["ModelSavePath"])

    yPred = predict(model=trainedModel, xTest=X_test_processed)

    results = evaluateModel(yPred, y_test)

    saveMetrics(results, configData["ResultsPath"])


