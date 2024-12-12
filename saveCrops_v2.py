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
        self.current_positive_points = []
        self.current_negative_points = []
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

            # Segmented images display with horizontal scroll bar only
        self.segmented_images_layout = QtWidgets.QHBoxLayout()
        segmented_images_widget = QtWidgets.QWidget()
        segmented_images_widget.setLayout(self.segmented_images_layout)

        # Set a fixed height for the widget holding the segmented images
        segmented_images_widget.setFixedHeight(200)  # Adjust the height as needed

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidget(segmented_images_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # Ensure the height of the scroll area matches the fixed height of the content
        self.scroll_area.setFixedHeight(200)

        layout.addWidget(self.scroll_area)



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
            labels_positive = [1] * len(points[0])  # Create labels for the current object
            labels_negative = [0] * len(points[1])  # Create labels for the current object
            points_formatted_positive = [list(point) for point in points[0]]  # Format points
            points_formatted_negative = [list(point) for point in points[1]]  # Format points
            print(f"Positive Points: {points_formatted_positive}")
            print(f"Negative Points: {points_formatted_negative}")
            
            print(f"Labels positive: {labels_positive}")
            print(f"Labels negative: {labels_negative}")


            # Run inference with SAM for each object
            results = self.model(self.image, points=[points_formatted_positive+points_formatted_negative], labels=[labels_positive+labels_negative])

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

            self.current_positive_points.append((scaled_x, scaled_y))
            # Draw a circle for the clicked point on the overlay
            cv2.circle(self.overlay, (scaled_x, scaled_y), 5, (0, 255, 0), -1)
            self.display_image(self.overlay)
            print(f"Object {self.object_counter}: Point selected - {scaled_x}, {scaled_y}")
            self.undo_stack.append(("mark_point", (scaled_x, scaled_y), self.all_points.copy(), self.segmented_images.copy()))  # Add action to undo stack
            self.redo_stack.clear()  # Clear redo stack after new action
            self.undo_action.setEnabled(True)
            self.redo_action.setEnabled(False)
            self.finalize_action.setEnabled(True)

            self.finalize_current_object(True)
            self.segment_with_points()
        elif event.button() == QtCore.Qt.RightButton:
            x = event.pos().x()
            y = event.pos().y()

            # Scale points to match the original image size
            scaled_x = int(x * self.image.shape[1] / self.image_label.width())
            scaled_y = int(y * self.image.shape[0] / self.image_label.height())

            self.current_negative_points.append((scaled_x, scaled_y))
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
            if self.current_positive_points:
                if len(self.all_points) < self.object_counter:
                    self.all_points.extend([[self.current_positive_points[:],[]]])
                else:
                    self.all_points[self.object_counter - 1][0] = self.current_positive_points[:]
                
            if self.current_negative_points:
                if len(self.all_points) < self.object_counter:
                    self.all_points.extend([[[],self.current_negative_points[:]]])
                    return
                self.all_points[self.object_counter - 1][1] = self.current_negative_points[:]
                return
            return

        if self.current_positive_points:
            self.all_points.append([self.current_positive_points[:],self.current_negative_points[:]])  # Save the current object's points
            print(f"Object {self.object_counter} points finalized: {self.current_positive_points}")
            self.undo_stack.append(("finalize", self.current_positive_points[:], self.all_points.copy(), self.segmented_images.copy()))  # Add action to undo stack
            self.redo_stack.clear()  # Clear redo stack after new action
            self.current_positive_points.clear()  # Clear for the next object
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

                    # Create a widget to display the segmented image and add delete/name/crop options
                    segment_widget = QtWidgets.QWidget(self)
                    segment_layout = QtWidgets.QVBoxLayout(segment_widget)

                    segment_label = QtWidgets.QLabel(self)
                    segment_label.setScaledContents(True)
                    segment_height, segment_width, segment_channel = imcrop_rgba.shape
                    bytes_per_line = segment_channel * segment_width
                    q_image = QtGui.QImage(cv2.cvtColor(imcrop_rgba, cv2.COLOR_BGRA2RGBA), segment_width, segment_height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    segment_label.setPixmap(pixmap)
                    segment_layout.addWidget(segment_label)

                    # Create a horizontal layout for name, crop, and delete options
                    options_layout = QtWidgets.QHBoxLayout()

                    # Add a line edit for naming the segment
                    name_edit = QtWidgets.QLineEdit(f"Segment {index + 1}", self)
                    name_edit.setPlaceholderText("Enter name for the segment")
                    options_layout.addWidget(name_edit)

                    # Add a crop button with an icon
                    crop_button = QtWidgets.QPushButton(qta.icon('mdi.crop', color='blue'), '', self)
                    crop_button.setToolTip("Crop Segment")
                    crop_button.clicked.connect(lambda _, i=index, label=segment_label: self.crop_segment(i, label))
                    options_layout.addWidget(crop_button)

                    # Add a delete button with an icon
                    delete_button = QtWidgets.QPushButton(qta.icon('fa.trash', color='red'), '', self)
                    delete_button.setToolTip("Delete Segment")
                    delete_button.clicked.connect(lambda _, i=index: self.delete_segment(i))
                    options_layout.addWidget(delete_button)

                    segment_layout.addLayout(options_layout)

                    self.segmented_images_layout.addWidget(segment_widget)

        # Display the final overlay with masks
        self.display_image(result_overlay)
        self.save_action.setEnabled(True)


    def crop_segment(self, index, label):
        if 0 <= index < len(self.segmented_images):
            segment_image = self.segmented_images[index]

            # Convert RGBA to BGR for cropping
            segment_image_bgr = cv2.cvtColor(segment_image, cv2.COLOR_BGRA2BGR)

            # Use OpenCV's ROI selection tool for cropping
            roi = cv2.selectROI("Crop Segment", segment_image_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Crop Segment")

            if roi[2] > 0 and roi[3] > 0:
                x, y, w, h = map(int, roi)
                cropped_image = segment_image[y:y+h, x:x+w]

                # Update the segmented image with the cropped version
                self.segmented_images[index] = cropped_image

                # Update the display in the segmented pane
                segment_height, segment_width, segment_channel = cropped_image.shape
                bytes_per_line = segment_channel * segment_width
                q_image = QtGui.QImage(cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2RGBA), segment_width, segment_height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                label.setPixmap(pixmap)

                print(f"Segment {index + 1} cropped successfully.")


    def delete_segment(self, index):
        if 0 <= index < len(self.segmented_images):
            del self.segmented_images[index]
            del self.current_results[index]
            self.all_points.pop(index)
            self.object_counter -= 1
            self.display_results(self.current_results)
            print(f"Deleted segment {index + 1}")


    def display_image(self, image):
        # Convert the image to RGB format if it's BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            # Draw all points on the image if they exist
        for segment in self.all_points:
            for point in segment[0]:
                x, y = point
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            for point in segment[1]:
                x, y = point
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)


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
            # Iterate through each segmented widget and save the corresponding image
            for idx in range(self.segmented_images_layout.count()):
                segment_widget = self.segmented_images_layout.itemAt(idx).widget()
                if not segment_widget:
                    continue

                # Get the segment name from the QLineEdit
                name_edit = segment_widget.findChild(QtWidgets.QLineEdit)
                segment_name = name_edit.text() if name_edit and name_edit.text().strip() else f"segment_{idx + 1}"

                # Ensure the name is valid for a file
                segment_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in segment_name).strip()

                # Save the corresponding image
                segment_image = self.segmented_images[idx]
                filename = f"{save_path}/{segment_name}.png"
                cv2.imwrite(filename, segment_image)
                print(f"Saved segment {idx + 1} as {filename}")


    def undo_action_triggered(self):
        if not self.undo_stack:
            return

        action, data, all_points_backup, segmented_images_backup = self.undo_stack.pop()
        self.redo_stack.append((action, data, all_points_backup, segmented_images_backup))
        self.redo_action.setEnabled(True)

        if action == "mark_point":
            if self.current_positive_points:
                self.current_positive_points.pop()
                # Redraw the overlay without the last point
                self.overlay = self.image.copy()
                for point in self.current_positive_points:
                    cv2.circle(self.overlay, point, 5, (0, 255, 0), -1)
                self.display_image(self.overlay)
        elif action == "finalize":
            if self.all_points:
                self.all_points.pop()
                self.object_counter -= 1
                self.current_positive_points.clear()
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
            self.current_positive_points.append(data)
            cv2.circle(self.overlay, data, 5, (0, 255, 0), -1)
            self.display_image(self.overlay)
        elif action == "finalize":
            self.all_points.append(self.current_positive_points[:])
            self.object_counter += 1
            self.current_positive_points.clear()
        elif action == "segment" or action == "auto_segment":
            self.display_results(data)

        if not self.redo_stack:
            self.redo_action.setEnabled(False)

    def reset_segments(self):
        # Clear all segments and reset the UI
        self.current_positive_points.clear()
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
