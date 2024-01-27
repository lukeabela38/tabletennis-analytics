import cv2, json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pose_estimation.processing import draw_landmarks_on_image
from pose_estimation.config import PoseEstimationConfig

### SETUP ###
model = PoseEstimationConfig.MODEL.value

base_options = python.BaseOptions(
    model_asset_path=model
    )

VisionRunningMode = mp.tasks.vision.RunningMode

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    output_segmentation_masks=PoseEstimationConfig.SEGMENTATION_MASKS.value)
detector = vision.PoseLandmarker.create_from_options(options)

### END OF SETUP ###

image = mp.Image.create_from_file("pose_estimation/images/inputs/topview.jpeg")
detection_result = detector.detect(image)

with open(PoseEstimationConfig.MAPPINGS.value) as file: 
    model_mappings = json.load(file)

results_to_print = (detection_result.pose_landmarks[0])
for i in range(len(results_to_print)):
    print(f"Category: {model_mappings[str(i)]}, X: {results_to_print[i].x}, Y: {results_to_print[i].y}, Z: {results_to_print[i].z}, Visibility: {results_to_print[i].visibility}, Presence: {results_to_print[i].presence}")

# STEP 5: Process the detection result. In this case, visualize it.
if PoseEstimationConfig.ANNOTATION.value:
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imwrite("pose_estimation/images/outputs/annotations/topview.jpeg", annotated_image) 

if PoseEstimationConfig.SEGMENTATION_MASKS.value:
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2.imwrite("pose_estimation/images/outputs/mask/topview.jpeg", visualized_mask) 
