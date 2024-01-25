import requests
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw

class ObjectDetection():
    def __init__(self, checkpoint: str = "google/owlvit-base-patch32"):
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.threshold = 0.1

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def processor(self):
        return self._processor

    @processor.setter
    def processor(self, processor):
        self._processor = processor

    def process(self, image, labels=[]):

        im = Image.open(image)
        inputs = self._processor(text=labels, images=im, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)
            target_sizes = torch.tensor([im.size[::-1]])
            results = self._processor.post_process_object_detection(outputs, threshold=self.threshold, target_sizes=target_sizes)[0]

        return results

    @staticmethod
    def draw(results, image, labels=[], save_path="images/outputs/test.jpg"):

        im = Image.open(image)
        draw = ImageDraw.Draw(im)

        scores = results["scores"].tolist()
        labels = results["labels"].tolist()
        boxes = results["boxes"].tolist()

        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{labels[label]}: {round(score,2)}", fill="white")

        im.save(save_path) 

        return save_path


def main():

    image = "images/inputs/topview.jpeg"
    labels=["person", "table", "net", "ball"]

    obj = ObjectDetection()

    results = obj.process(image=image, labels=labels)

    obj.draw(results=results, image=image, labels=labels, save_path="images/outputs/topview.jpeg")


if __name__ == "__main__":
    main()