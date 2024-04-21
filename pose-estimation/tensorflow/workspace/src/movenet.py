import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.helpers import draw_prediction_on_image, to_gif
from src.cropping import init_crop_region, run_inference, determine_crop_region

class MOVENET():
    def __init__(self, model_path="artifacts/models/model.tflite", input_size = 192):

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_size = input_size

    def movenet(self, input_image):
        """Runs detection on an input image.

        Args:
            input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
            A [1, 1, 17, 3] float numpy array representing the predicted keypoint
            coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        self.interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores
    
    def movenet_image(self, input_image_path: str, visualize: bool, output_image_path: str):
        image = tf.io.read_file(input_image_path)
        image = tf.image.decode_jpeg(image)

        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        keypoints_with_scores = self.movenet(input_image)

        if visualize:
            self.visualise_image(
                image=image, 
                keypoints_with_scores=keypoints_with_scores, 
                output_image_path=output_image_path
                )

        return keypoints_with_scores

    @staticmethod
    def visualise_image(image, keypoints_with_scores, output_image_path="artifacts/outputs/output_image.jpeg"):
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

        plt.figure(figsize=(10, 10))
        plt.imsave(output_image_path, output_overlay)

        return None
    
    def movenet_video(self, input_video_path: str, visualize: bool, output_video_path: str):
        image = tf.io.read_file(input_video_path)
        image = tf.image.decode_gif(image)

        # Load the input image.
        num_frames, image_height, image_width, _ = image.shape
        crop_region = init_crop_region(image_height, image_width)

        output_images = []
        for frame_idx in range(num_frames):
            keypoints_with_scores = run_inference(
                self.movenet, image[frame_idx, :, :, :], crop_region,
                crop_size=[self.input_size, self.input_size]
                )
            output_images.append(draw_prediction_on_image(
                image[frame_idx, :, :, :].numpy().astype(np.int32),
                keypoints_with_scores, crop_region=None,
                close_figure=True, output_image_height=300)
                )
            crop_region = determine_crop_region(
                keypoints_with_scores, image_height, image_width)

        if visualize:
            output = np.stack(output_images, axis=0)
            to_gif(output, duration=100, output_video_path=output_video_path)

        return keypoints_with_scores