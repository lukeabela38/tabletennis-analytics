from enum import Enum

class PoseEstimationConfig(Enum):
    SEGMENTATION_MASKS: bool = False
    ANNOTATION: bool = False
    MODEL: str = "pose_landmarker.task"
