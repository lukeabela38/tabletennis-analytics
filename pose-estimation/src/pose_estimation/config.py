from enum import Enum

class PoseEstimationConfig(Enum):
    MODEL: str = "/src/pose_estimation/artifacts/models/pose_landmarker.task"
    MAPPINGS: str = "/src/pose_estimation/artifacts/models/mappings.json"
