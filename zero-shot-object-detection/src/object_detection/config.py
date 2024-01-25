from enum import Enum

class ObjectDetectionConfig(Enum):
    CHECKPOINT: str = "google/owlvit-base-patch32"
    LABELS: list =["person", "table", "net", "ball"]
    THRESHOLD: float = 0.1