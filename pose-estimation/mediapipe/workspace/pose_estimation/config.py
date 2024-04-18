from enum import Enum

class PoseEstimationConfig(Enum):
    MODEL: str = "/src/pose_estimation/models/pose_landmarker.task"
    MAPPINGS: str = "/src/pose_estimation/models/mappings.json"
    ARTIFACTS: str = "/src/pose_estimation/artifacts/"
