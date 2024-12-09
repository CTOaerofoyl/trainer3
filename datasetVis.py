import os
import glob
import cv2


def visualize_labels(image_path, label_path):
    """
    Visualize bounding boxes on the corresponding image.

    :param image_path: Path to the directory containing images.
    :param label_path: Path to the directory containing label files.
    """
    label_files = glob.glob(os.path.join(label_path, '*.txt'))
    cv2.namedWindow("Bounding Boxes", cv2.WINDOW_NORMAL)

    for label_file in label_files:
        image_file = os.path.join(image_path, os.path.splitext(os.path.basename(label_file))[0] + ".jpg")

        if not os.path.exists(image_file):
            print(f"Image file not found for label file: {label_file}")
            continue

        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to read image: {image_file}")
            continue

        with open(label_file, 'r') as file:
            lines = file.readlines()

        height, width, _ = image.shape

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping malformed line in {label_file}: {line}")
                continue

            class_id, x_center, y_center, box_width, box_height = map(float, parts)

            # Convert normalized coordinates back to pixel coordinates
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            box_width = int(box_width * width)
            box_height = int(box_height * height)

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image

        cv2.imshow("Bounding Boxes", image)
        cv2.waitKey(0)

# Define paths for train and val label directories
dataset1_path = "dataset7"
train_labels_path = f"{dataset1_path}/labels/train"
val_labels_path = f"{dataset1_path}/labels/val"
train_images_path = f"{dataset1_path}/images/train"
val_images_path = f"{dataset1_path}/images/val"

# Visualize the labels
visualize_labels(train_images_path, train_labels_path)
visualize_labels(val_images_path, val_labels_path)

