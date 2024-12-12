### File: app/image_display.py
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np

class ImageDisplayHandler:
    def __init__(self, parent):
        self.parent = parent
        self.scroll_area = QtWidgets.QScrollArea()

    def create_segmented_images_layout(self):
        segmented_images_widget = QtWidgets.QWidget()
        segmented_images_layout = QtWidgets.QHBoxLayout(segmented_images_widget)

        segmented_images_widget.setFixedHeight(200)
        self.scroll_area.setWidget(segmented_images_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFixedHeight(200)

        return segmented_images_layout

    def update_segmented_images(self, segmented_images):
        layout = self.scroll_area.widget().layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for idx, segment_image in enumerate(segmented_images):
            height, width, channel = segment_image.shape
            bytes_per_line = channel * width
            q_image = QtGui.QImage(segment_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            label = QtWidgets.QLabel()
            label.setPixmap(pixmap)
            layout.addWidget(label)

    def display_overlay(self, image, points):
        overlay = image.copy()
        for segment in points:
            for point in segment[0]:
                x, y = point
                cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
            for point in segment[1]:
                x, y = point
                cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)

        return overlay
