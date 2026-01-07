from pipelines.training_pipeline import TrainingPipeline
import yaml
import mlflow

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    mlflow.set_experiment("disaster-prediction v1")
    config = load_config()
    pipeline_instance = TrainingPipeline(configData=config)
    pipeline_instance.run()
