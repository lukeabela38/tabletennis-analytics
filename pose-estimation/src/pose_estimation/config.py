from enum import Enum

class PoseEstimationConfig(Enum):
    SEGMENTATION_MASKS: bool = False
    ANNOTATION: bool = False
    MODEL: str = "/src/pose_estimation/artifacts/models/pose_landmarker.task"
    MAPPINGS: str = "/src/pose_estimation/artifacts/models/mappings.json"