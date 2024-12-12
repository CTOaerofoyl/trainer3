### File: app/segmentation.py
from ultralytics import SAM

class SegmentationHandler:
    def __init__(self):
        self.model = SAM("sam_b.pt")

    def run_inference(self, image, points, labels):
        return self.model(image, points=points, labels=labels)

    def auto_segment(self, image):
        return self.model(image)