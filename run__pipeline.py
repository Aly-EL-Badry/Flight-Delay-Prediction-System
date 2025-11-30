from pipelines.training_pipeline import TrainingPipeline
import yaml

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.runtime_version")

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    pipeline_instance = TrainingPipeline(configData=config)
    pipeline_instance.run()
