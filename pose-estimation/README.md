# Pose Estimation

## To run

```bash
docker build -t pose_estimation .
xhost +
docker run -it --rm -v ./src:/src -p 8000:8000 --device=/dev/video2 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix pose_estimation
```