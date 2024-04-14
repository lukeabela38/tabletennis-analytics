# Breakdown of data

## Pose Landmarks

[here](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#get_started)

A list of pose landmarks. Each landmark consists of the following:

- x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
- z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
- visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.


##