from src.movenet import MOVENET

# Load the input image.
input_size = 192
model_path = "artifacts/models/model.tflite"
input_image_path = 'artifacts/inputs/input_image.jpeg'
output_image_path = 'artifacts/outputs/output_image.jpeg'
visualize=True

movenet = MOVENET(model_path=model_path, input_size=input_size)

# Run model inference.
keypoints_with_scores = movenet.movenet_image(
    input_image_path=input_image_path,
    visualize=visualize,
    output_image_path=output_image_path
    )

print(keypoints_with_scores)

input_video_path = 'artifacts/inputs/input_gif.gif'
output_video_path = 'artifacts/outputs/output_gif.gif'

# Run model inference.
keypoints_with_scores = movenet.movenet_video(
    input_video_path=input_video_path,
    visualize=visualize,
    output_video_path=output_video_path
    )

print(keypoints_with_scores)