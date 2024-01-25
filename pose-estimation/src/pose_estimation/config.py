from enum import Enum

class PoseEstimationConfig(Enum):
    SEGMENTATION_MASKS: bool = True
    ANNOTATION: bool = True
    MODEL: str = "pose_landmarker.task"
