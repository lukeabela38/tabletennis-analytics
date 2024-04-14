import json, cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from pose_estimation.config import PoseEstimationConfig
from datetime import datetime

DIRECTORY: str = PoseEstimationConfig.ARTIFACTS.value
with open(PoseEstimationConfig.MAPPINGS.value) as file: 
    model_mappings = json.load(file)

class LiveStreamPoseDetection():
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
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.interpret_results
            )
        self._model = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def classify(self):
        with self._model as landmarker:
            # Use OpenCV’s VideoCapture to load the input video.
            cap = cv2.VideoCapture(-1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_index: float = 0
            # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
            # You’ll need it to calculate the timestamp for each frame.
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret==True:
                    # Loop through each frame in the video using VideoCapture#read()
                    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    numpy_frame_from_opencv = np.asarray(image)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
                    
                    # Perform pose landmarking on the provided single image.
                    # The pose landmarker must be created with the video mode.
                    frame_timestamp_ms = int(1000 * (frame_index/fps))
                    frame_index += 1
                    pose_landmarker_result = landmarker.detect_async(mp_image, frame_timestamp_ms)

                    #print(pose_landmarker_result.pose_landmarks[0])
                else:
                    cap.release()
                    break

    @staticmethod
    def interpret_results(result, output_image: mp.Image, timestamp_ms: int):
        results_to_print = (result.pose_landmarks[0])
        dict = {}
        for i in range(len(results_to_print)):
            nested_dict = {}
            nested_dict["x"] = results_to_print[i].x
            nested_dict["y"] = results_to_print[i].y
            nested_dict["z"] = results_to_print[i].z
            nested_dict["visibility"] = results_to_print[i].visibility
            nested_dict["presence"] = results_to_print[i].presence

            dict["timestamp"] = datetime.now().strftime("%Y%m%d-%H%M%S")
            dict[model_mappings[str(i)]] = nested_dict

        filepath = f"{DIRECTORY}livestream.json"
        with open(filepath, "w") as outfile:
            json.dump(dict, outfile)

        return None