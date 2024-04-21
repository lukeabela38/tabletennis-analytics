# Pose Estimation

## To run

```bash
docker build -t pose_estimation_tf .
xhost +
docker run -it --rm -v $(pwd)/workspace:/src -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pose_estimation_tf
```

