from object_detection.zero_shot_class import ObjectDetection

def main():

    image = "images/inputs/behind.jpeg"

    obj = ObjectDetection()
    results = obj.classify(image=image)
    #obj.draw(results=results, image=image, save_path="images/outputs/behind.jpeg")


if __name__ == "__main__":
    main()