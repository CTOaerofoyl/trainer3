import sys
import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets, QtGui, QtCore
from ultralytics import SAM
import qtawesome as qta

class SegmentationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Initialize variables
        self.image = None
        self.overlay = None
        self.current_points = []
        self.all_points = []
        self.object_counter = 1
        self.model = SAM("sam_b.pt")  # Load SAM model
        self.segmented_images = []  # Store segmented images
        self.undo_stack = []  # Stack to store undo actions
        self.redo_stack = []  # Stack to store redo actions
        self.current_results = []  # Store current segmentation results

    def initUI(self):
        # Main widget and layout setup
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Stream URL input
        self.url_input = QtWidgets.QLineEdit(self)
        self.url_input.setPlaceholderText('Enter RTSP Stream URL here...')
        self.url_input.setText('rtsp://admin:PAG00319@192.168.1.223:554/live')
        layout.addWidget(self.url_input)

        # Toolbar Setup
        self.toolbar = QtWidgets.QToolBar("Toolbar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        # Capture button
        self.capture_action = QtWidgets.QAction(qta.icon('fa.camera'), '', self)
        self.capture_action.setToolTip('Capture Image')
        self.capture_action.triggered.connect(self.capture_image)
        self.toolbar.addAction(self.capture_action)

        # Segment button (uses selected points)
        self.segment_action = QtWidgets.QAction(qta.icon('ri.scissors-cut-fill'), '', self)
        self.segment_action.setToolTip('Segment with Selected Points')
        self.segment_action.triggered.connect(self.segment_with_points)
        self.segment_action.setEnabled(False)
        self.toolbar.addAction(self.segment_action)

        # Auto-segment button
        self.autosegment_action = QtWidgets.QAction(qta.icon('fa.magic'), '', self)
        self.autosegment_action.setToolTip('Auto-Segment')
        self.autosegment_action.triggered.connect(self.auto_segment)
        self.autosegment_action.setEnabled(False)
        self.toolbar.addAction(self.autosegment_action)

        # Mark points button
        self.mark_points_action = QtWidgets.QAction(qta.icon('mdi.selection-ellipse-arrow-inside'), '', self)
        self.mark_points_action.setToolTip('Mark Points')
        self.mark_points_action.triggered.connect(self.mark_points)
        self.mark_points_action.setEnabled(False)
        self.toolbar.addAction(self.mark_points_action)

        # Finalize object button
        self.finalize_action = QtWidgets.QAction(qta.icon('fa.check'), '', self)
        self.finalize_action.setToolTip('Finalize Object')
        self.finalize_action.triggered.connect(self.finalize_current_object)
        self.finalize_action.setEnabled(False)
        self.toolbar.addAction(self.finalize_action)

        # Save button
        self.save_action = QtWidgets.QAction(qta.icon('fa.save'), '', self)
        self.save_action.setToolTip('Save Segments')
        self.save_action.triggered.connect(self.save_segments)
        self.save_action.setEnabled(False)
        self.toolbar.addAction(self.save_action)

        # Undo button
        self.undo_action = QtWidgets.QAction(qta.icon('mdi.undo-variant'), '', self)
        self.undo_action.setToolTip('Undo Last Action')
        self.undo_action.triggered.connect(self.undo_action_triggered)
        self.undo_action.setEnabled(False)
        self.toolbar.addAction(self.undo_action)

        # Redo button
        self.redo_action = QtWidgets.QAction(qta.icon('mdi.redo-variant'), '', self)
        self.redo_action.setToolTip('Redo Last Action')
        self.redo_action.triggered.connect(self.redo_action_triggered)
        self.redo_action.setEnabled(False)
        self.toolbar.addAction(self.redo_action)

        # Reset button
        self.reset_action = QtWidgets.QAction(qta.icon('fa.refresh'), '', self)
        self.reset_action.setToolTip('Reset All Segments')
        self.reset_action.triggered.connect(self.reset_segments)
        self.toolbar.addAction(self.reset_action)

        # Image display
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setScaledContents(True)  # Allow scaling of the image
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.get_points  # Set mouse event to get_points method initially
        layout.addWidget(self.image_label)

        # Segmented images display
        self.segmented_images_layout = QtWidgets.QHBoxLayout()
        self.segmented_images_group = QtWidgets.QGroupBox("Segmented Objects")
        self.segmented_images_group.setLayout(self.segmented_images_layout)
        self.segmented_images_group.setMaximumHeight(200)  # Limit the height of the segmented objects pane
        layout.addWidget(self.segmented_images_group)

        # Set central widget
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
            self.display_image(self.image)
            self.segment_action.setEnabled(True)
            self.autosegment_action.setEnabled(True)
            self.mark_points_action.setEnabled(True)
            self.finalize_action.setEnabled(True)
            self.save_action.setEnabled(True)  # Enable save button after capture
            self.mark_points = False
        else:
            QtWidgets.QMessageBox.critical(self, 'Error', 'Failed to capture image. Check the URL and try again.')

    def segment_with_points(self):
        if self.image is None or not self.all_points:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please select points for at least one object before segmenting.')
            return
        
        # Prepare data for SAM inference
        aggregated_results = []

        for points in self.all_points:
            labels = [1] * len(points)  # Create labels for the current object
            points_formatted = [list(point) for point in points]  # Format points
            print(f"Points: {points_formatted}")
            print(f"Labels: {labels}")

            # Run inference with SAM for each object
            results = self.model(self.image, points=[points_formatted], labels=[labels])
            aggregated_results.extend(results)

        self.current_results = aggregated_results
        self.display_results(aggregated_results)
        self.undo_stack.append(("segment", aggregated_results, self.all_points.copy(), self.segmented_images.copy()))  # Add action to undo stack
        self.redo_stack.clear()  # Clear redo stack after new action
        self.undo_action.setEnabled(True)
        self.redo_action.setEnabled(False)

    def auto_segment(self):
        if self.image is None:
            return

        # Run SAM model to automatically segment the image
        results = self.model(self.image)
        self.current_results = results
        self.display_results(results)
        self.undo_stack.append(("auto_segment", results, self.all_points.copy(), self.segmented_images.copy()))  # Add action to undo stack
        self.redo_stack.clear()  # Clear redo stack after new action
        self.undo_action.setEnabled(True)
        self.redo_action.setEnabled(False)

    def mark_points(self):
        if self.image is None:
            return
        self.mark_points = not self.mark_points
        

    def get_points(self, event):
        if not self.mark_points:  # Check if mark points button is enabled
            return  # Do nothing if marking is not enabled

        if event.button() == QtCore.Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()

            # Scale points to match the original image size
            scaled_x = int(x * self.image.shape[1] / self.image_label.width())
            scaled_y = int(y * self.image.shape[0] / self.image_label.height())

            self.current_points.append((scaled_x, scaled_y))
            # Draw a circle for the clicked point on the overlay
            cv2.circle(self.overlay, (scaled_x, scaled_y), 5, (0, 255, 0), -1)
            self.display_image(self.overlay)
            print(f"Object {self.object_counter}: Point selected - {scaled_x}, {scaled_y}")
            self.undo_stack.append(("mark_point", (scaled_x, scaled_y), self.all_points.copy(), self.segmented_images.copy()))  # Add action to undo stack
            self.redo_stack.clear()  # Clear redo stack after new action
            self.undo_action.setEnabled(True)
            self.redo_action.setEnabled(False)
            self.finalize_current_object(True)
            self.segment_with_points()

    def finalize_current_object(self, temporary=False):
        if temporary:
            if self.current_points:
                if len(self.all_points) < self.object_counter:
                    self.all_points.extend([self.current_points[:]])
                    return
                self.all_points[self.object_counter - 1] = self.current_points[:]
                return

        if self.current_points:
            self.all_points.append(self.current_points[:])  # Save the current object's points
            print(f"Object {self.object_counter} points finalized: {self.current_points}")
            self.undo_stack.append(("finalize", self.current_points[:], self.all_points.copy(), self.segmented_images.copy()))  # Add action to undo stack
            self.redo_stack.clear()  # Clear redo stack after new action
            self.current_points.clear()  # Clear for the next object
            self.object_counter += 1  # Increment the object counter
            self.undo_action.setEnabled(True)
            self.redo_action.setEnabled(False)
        else:
            print("No points selected for the current object!")

    def clearSegmentedImages(self):
        while self.segmented_images_layout.count():
            child = self.segmented_images_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def display_results(self, results):
        # Create a copy of the original image to overlay segmentation results
        result_overlay = self.image.copy()
        self.clearSegmentedImages()
        # Overlay segmentation results with different colors for each object
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Predefined colors
        color_index = 0

        for index, result in enumerate(results):
            if hasattr(result, 'masks') and result.masks.data is not None:
                for mask_tensor in result.masks.data:
                    mask_array = mask_tensor.cpu().numpy().astype(np.uint8)
                    color = colors[color_index % len(colors)]
                    color_index += 1

                    # Create a translucent mask overlay
                    for c in range(3):
                        result_overlay[..., c] = np.where(mask_array == 1, 
                                                          (0.5 * result_overlay[..., c] + 0.5 * color[c]).astype(np.uint8),
                                                          result_overlay[..., c])

                    # Crop the segmented region from the original image and apply the mask for transparency
                    x1, y1, x2, y2 = [int(i) for i in result.boxes.xyxy[0].cpu()]
                    x1 = max(0, min(x1, self.image.shape[1]))
                    x2 = max(0, min(x2, self.image.shape[1]))
                    y1 = max(0, min(y1, self.image.shape[0]))
                    y2 = max(0, min(y2, self.image.shape[0]))
                    imcrop = self.image[y1:y2, x1:x2]
                    mask_crop = mask_array[y1:y2, x1:x2]

                    # Create an RGBA image with transparency for non-object pixels
                    imcrop_rgba = cv2.cvtColor(imcrop, cv2.COLOR_BGR2BGRA)
                    imcrop_rgba[:, :, 3] = np.where(mask_crop == 1, 255, 0)
                    if len(self.segmented_images) - 1 < index:
                        self.segmented_images.append(imcrop_rgba.copy())
                    else:
                        self.segmented_images[index] = (imcrop_rgba.copy())
                    # Display each segmented object on the side without changing its colors
                    segment_label = QtWidgets.QLabel(self)
                    segment_label.setScaledContents(True)
                    segment_height, segment_width, segment_channel = imcrop_rgba.shape
                    bytes_per_line = segment_channel * segment_width
                    q_image = QtGui.QImage(cv2.cvtColor(imcrop_rgba, cv2.COLOR_BGRA2RGBA), segment_width, segment_height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    segment_label.setPixmap(pixmap)
                    self.segmented_images_layout.addWidget(segment_label)

        # Display the final overlay with masks
        self.display_image(result_overlay)

    def display_image(self, image):
        # Convert the image to RGB format if it's BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        # Resize image to fit within the QLabel while maintaining aspect ratio
        max_width = self.image_label.width()
        max_height = self.image_label.height()
        h, w = image.shape[:2]
        aspect_ratio = w / h

        if w > max_width or h > max_height:
            if aspect_ratio > 1:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        height, width, channel = image.shape
        bytes_per_line = channel * width
        q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888 if channel == 3 else QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def save_segments(self):
        save_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory to Save Segments')
        if save_path:
            for idx, segment_image in enumerate(self.segmented_images):
                filename = f"{save_path}/segment_{idx + 1}.png"
                cv2.imwrite(filename, segment_image)
                print(f"Saved segment {idx + 1} as {filename}")

    def undo_action_triggered(self):
        if not self.undo_stack:
            return

        action, data, all_points_backup, segmented_images_backup = self.undo_stack.pop()
        self.redo_stack.append((action, data, all_points_backup, segmented_images_backup))
        self.redo_action.setEnabled(True)

        if action == "mark_point":
            if self.current_points:
                self.current_points.pop()
                # Redraw the overlay without the last point
                self.overlay = self.image.copy()
                for point in self.current_points:
                    cv2.circle(self.overlay, point, 5, (0, 255, 0), -1)
                self.display_image(self.overlay)
        elif action == "finalize":
            if self.all_points:
                self.all_points.pop()
                self.object_counter -= 1
                self.current_points.clear()
        elif action == "segment" or action == "auto_segment":
            # Restore the segmented images and points
            self.all_points = all_points_backup
            self.segmented_images = segmented_images_backup
            self.display_results(self.segmented_images)

        if not self.undo_stack:
            self.undo_action.setEnabled(False)

    def redo_action_triggered(self):
        if not self.redo_stack:
            return

        action, data, all_points_backup, segmented_images_backup = self.redo_stack.pop()
        self.undo_stack.append((action, data, all_points_backup, segmented_images_backup))
        self.undo_action.setEnabled(True)

        if action == "mark_point":
            self.current_points.append(data)
            cv2.circle(self.overlay, data, 5, (0, 255, 0), -1)
            self.display_image(self.overlay)
        elif action == "finalize":
            self.all_points.append(self.current_points[:])
            self.object_counter += 1
            self.current_points.clear()
        elif action == "segment" or action == "auto_segment":
            self.display_results(data)

        if not self.redo_stack:
            self.redo_action.setEnabled(False)

    def reset_segments(self):
        # Clear all segments and reset the UI
        self.current_points.clear()
        self.all_points.clear()
        self.overlay = self.image.copy() if self.image is not None else None
        self.segmented_images.clear()
        self.current_results.clear()
        self.object_counter = 1
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        self.segment_action.setEnabled(False)
        self.autosegment_action.setEnabled(False)
        self.mark_points_action.setEnabled(False)
        self.finalize_action.setEnabled(False)
        self.save_action.setEnabled(False)

        # Clear the segmented images layout
        for i in reversed(range(self.segmented_images_layout.count())):
            widget = self.segmented_images_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if self.image is not None:
            self.display_image(self.image)
        print("All segments have been reset.")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = SegmentationApp()
    ex.show()
    sys.exit(app.exec_())
