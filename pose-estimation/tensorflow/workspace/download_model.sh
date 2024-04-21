#!/bin/bash

wget -q -O models/model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite

curl -o inputs/input_image.jpeg https://images.pexels.com/photos/4384679/pexels-photo-4384679.jpeg --silent
