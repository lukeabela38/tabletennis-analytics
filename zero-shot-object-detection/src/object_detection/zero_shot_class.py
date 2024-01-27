import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw
from object_detection.config import ObjectDetectionConfig

class ObjectDetection():
    def __init__(self):
        checkpoint: str = ObjectDetectionConfig.CHECKPOINT.value
        self.labels: list = ObjectDetectionConfig.LABELS.value
        self.threshold = ObjectDetectionConfig.THRESHOLD.value

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self, labels):
        self._labels = labels
    
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

    def classify(self, image):

        im = Image.open(image)
        inputs = self._processor(text=self._labels, images=im, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)
            target_sizes = torch.tensor([im.size[::-1]])
            results = self._processor.post_process_object_detection(outputs, threshold=self.threshold, target_sizes=target_sizes)[0]

        return results

    @staticmethod 
    def logging(results):
        return results

    @staticmethod
    def draw(results, image, save_path="images/outputs/test.jpg"):

        im = Image.open(image)
        draw = ImageDraw.Draw(im)

        scores = results["scores"].tolist()
        labels = results["labels"].tolist()
        boxes = results["boxes"].tolist()

        for i in range(len(scores)):
            xmin, ymin, xmax, ymax = boxes[i]
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{labels[i]}: {round(scores[i],2)}", fill="white")

        im.save(save_path) 

        return save_path