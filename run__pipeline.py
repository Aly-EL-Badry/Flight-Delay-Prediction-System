from .pipelines.training_pipeline import TrainingPipeline
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    pipeline_instance = TrainingPipeline(config)
    pipeline_instance.run()
