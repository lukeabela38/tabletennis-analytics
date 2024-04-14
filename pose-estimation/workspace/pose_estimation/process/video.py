import json, cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from pose_estimation.config import PoseEstimationConfig
from pose_estimation.visualise import draw_landmarks_on_image
from datetime import datetime

DIRECTORY: str = PoseEstimationConfig.ARTIFACTS.value
with open(PoseEstimationConfig.MAPPINGS.value) as file: 
    model_mappings = json.load(file)

class VideoPoseDetection():
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
            running_mode=mp.tasks.vision.RunningMode.VIDEO
            )
        self._model = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def classify(self, input_path="pose_estimation/videos/inputs/forehand_drive_malong.mp4", output_path = "pose_estimation/videos/outputs/forehand_drive_malong.mp4"):
        with self._model as landmarker:
            # Use OpenCV’s VideoCapture to load the input video.
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2. CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2. CAP_PROP_FRAME_HEIGHT))

            encoding = cv2.VideoWriter_fourcc(*'mp4v')
            print(encoding)
            output_vid = cv2.VideoWriter(output_path, encoding, fps, (width, height))
            
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
                    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
                    cv_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    output_vid.write(cv_image)
                    self.interpret_results(pose_landmarker_result, frame_index, frame_timestamp_ms)
                else:
                    cap.release()
                    output_vid.release()
                    break

    @staticmethod
    def interpret_results(result, index, frame_timestamp_ms):

        results_to_print = (result.pose_landmarks[0])
        dict = {}

        for i in range(len(results_to_print)):
            nested_dict = {}
            nested_dict["x"] = results_to_print[i].x
            nested_dict["y"] = results_to_print[i].y
            nested_dict["z"] = results_to_print[i].z
            nested_dict["visibility"] = results_to_print[i].visibility
            nested_dict["presence"] = results_to_print[i].presence
            dict[model_mappings[str(i)]] = nested_dict

        dict["timestamp"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        filepath = f"{DIRECTORY}zhang_jike/livestream_{index}.json"

        with open(filepath, "w") as outfile:
            json.dump(dict, outfile)

        return None