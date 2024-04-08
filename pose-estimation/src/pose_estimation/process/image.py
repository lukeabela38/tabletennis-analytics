import json
import mediapipe as mp
from mediapipe.tasks import python
from pose_estimation.config import PoseEstimationConfig

with open(PoseEstimationConfig.MAPPINGS.value) as file: 
    model_mappings = json.load(file)

class ImagePoseDetection():
    def __init__(self):
        ### SETUP ###
        self.model = PoseEstimationConfig.MODEL.value

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        base_options = python.BaseOptions(
            model_asset_path=model
            )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
            )
        self._model = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def classify(self, path="pose_estimation/images/inputs/topview.jpeg"):
        image = mp.Image.create_from_file(path)
        return self._model.detect(image)

    @staticmethod
    def interpret_results(detection_result):
        results_to_print = (detection_result.pose_landmarks[0])
        for i in range(len(results_to_print)):
            print(f"Category: {model_mappings[str(i)]}, X: {results_to_print[i].x}, Y: {results_to_print[i].y}, Z: {results_to_print[i].z}, Visibility: {results_to_print[i].visibility}, Presence: {results_to_print[i].presence}")