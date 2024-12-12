### File: app/segmentation_app.py
from PyQt5 import QtWidgets, QtGui, QtCore
from segmenterApp.toolbar import Toolbar
from segmenterApp.segmentation import SegmentationHandler
from segmenterApp.image_display import ImageDisplayHandler
import cv2

class SegmentationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.image = None
        self.overlay = None
        self.current_positive_points = []
        self.current_negative_points = []
        self.all_points = []
        self.object_counter = 1
        self.segmented_images = []
        self.undo_stack = []
        self.redo_stack = []
        self.current_results = []

        # Handlers
        self.segmentation_handler = SegmentationHandler()
        self.image_display_handler = None  # Placeholder until initialized in initUI

        self.initUI()

    def initUI(self):
        # Main widget and layout setup
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Stream URL input
        self.url_input = QtWidgets.QLineEdit(self)
        self.url_input.setPlaceholderText('Enter RTSP Stream URL here...')
        self.url_input.setText('rtsp://admin:PAG00319@192.168.1.223:554/live')
        layout.addWidget(self.url_input)

        # Toolbar setup
        self.toolbar = Toolbar(self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        # Image display
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setScaledContents(True)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.get_points
        layout.addWidget(self.image_label)

        # Initialize ImageDisplayHandler now that the main UI is partially set up
        self.image_display_handler = ImageDisplayHandler(self)

        # Segmented images display
        self.segmented_images_layout = self.image_display_handler.create_segmented_images_layout()
        layout.addWidget(self.image_display_handler.scroll_area)

        self.setCentralWidget(central_widget)
        self.setWindowTitle('Segmentation Tool')
        self.setGeometry(100, 100, 1200, 800)

    def capture_image(self):
        url = self.url_input.text()
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.image = frame
            self.overlay = frame.copy()
            self.image_display_handler.display_overlay(self.image, self.all_points)
            self.toolbar.update_action_states(capture=True)
        else:
            QtWidgets.QMessageBox.critical(self, 'Error', 'Failed to capture image. Check the URL and try again.')

    def auto_segment(self):
        if self.image is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No image available for auto-segmentation.')
            return

        results = self.segmentation_handler.auto_segment(self.image)
        self.current_results = results
        self.image_display_handler.update_segmented_images(self.segmented_images)
        self.undo_stack.append(("auto_segment", results, self.all_points.copy(), self.segmented_images.copy()))
        self.redo_stack.clear()
        self.toolbar.update_action_states(capture=True)

    def segment_with_points(self):
        if self.image is None or not self.all_points:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select points for at least one object before segmenting.')
            return

        aggregated_results = []
        for points in self.all_points:
            positive_points = [list(point) for point in points[0]]
            negative_points = [list(point) for point in points[1]]

            labels_positive = [1] * len(positive_points)
            labels_negative = [0] * len(negative_points)

            results = self.segmentation_handler.run_inference(self.image, 
                                                              positive_points + negative_points, 
                                                              labels_positive + labels_negative)
            aggregated_results.append(results)

        self.current_results = aggregated_results
        self.image_display_handler.update_segmented_images(self.segmented_images)
        self.undo_stack.append(("segment", aggregated_results, self.all_points.copy(), self.segmented_images.copy()))
        self.redo_stack.clear()
        self.toolbar.update_action_states(capture=True)

    def get_points(self, event):
        if self.image is None:
            return

        x = int(event.pos().x() * self.image.shape[1] / self.image_label.width())
        y = int(event.pos().y() * self.image.shape[0] / self.image_label.height())

        if event.button() == QtCore.Qt.LeftButton:
            self.current_positive_points.append((x, y))
        elif event.button() == QtCore.Qt.RightButton:
            self.current_negative_points.append((x, y))

        self.overlay = self.image_display_handler.display_overlay(self.image, self.all_points)
        self.undo_stack.append(("mark", (x, y), self.all_points.copy(), self.segmented_images.copy()))

    def finalize_current_object(self):
        if self.current_positive_points or self.current_negative_points:
            self.all_points.append([self.current_positive_points.copy(), self.current_negative_points.copy()])
            self.current_positive_points.clear()
            self.current_negative_points.clear()
            self.object_counter += 1
            self.toolbar.update_action_states(capture=True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No points to finalize for the current object.')

    def reset_segments(self):
        self.image = None
        self.overlay = None
        self.current_positive_points.clear()
        self.current_negative_points.clear()
        self.all_points.clear()
        self.segmented_images.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.toolbar.update_action_states(capture=False)
        self.image_label.clear()

    def save_segments(self):
        save_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory to Save Segments')
        if save_path:
            for idx, segment in enumerate(self.segmented_images):
                filename = f"{save_path}/segment_{idx + 1}.png"
                cv2.imwrite(filename, segment)
                print(f"Saved segment {idx + 1} as {filename}")

    def undo_action_triggered(self):
        if not self.undo_stack:
            return

        action, data, all_points_backup, segmented_images_backup = self.undo_stack.pop()
        self.redo_stack.append((action, data, all_points_backup, segmented_images_backup))

        if action == "mark":
            if self.current_positive_points:
                self.current_positive_points.pop()
            elif self.current_negative_points:
                self.current_negative_points.pop()
        elif action == "segment" or action == "auto_segment":
            self.all_points = all_points_backup
            self.segmented_images = segmented_images_backup

        self.toolbar.update_action_states(capture=bool(self.image))

    def redo_action_triggered(self):
        if not self.redo_stack:
            return

        action, data, all_points_backup, segmented_images_backup = self.redo_stack.pop()
        self.undo_stack.append((action, data, all_points_backup, segmented_images_backup))

        if action == "mark":
            if data in self.current_positive_points:
                self.current_positive_points.append(data)
            else:
                self.current_negative_points.append(data)
        elif action == "segment" or action == "auto_segment":
            self.all_points = all_points_backup
            self.segmented_images = segmented_images_backup

        self.toolbar.update_action_states(capture=bool(self.image))
