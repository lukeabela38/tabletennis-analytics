# Ball Detection

## To run

```bash
docker build -t ball_detector .
docker run -it --rm -v ./src:/src ball_detector
```

## Methodology

1. Flip image to Gray Scale
2. Use HoughCircles Transform to extract circular objects within the image
3. Use distance and colour to decide which circles are the most appropriate
