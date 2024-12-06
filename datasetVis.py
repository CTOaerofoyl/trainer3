import os
import cv2

dataset_path = 'dataset1/images/train'
labels_path = 'dataset1/labels/train'
output_path = 'output_images'

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Get list of images in the dataset
image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

for image_file in image_files:
    # Load the image
    image_path = os.path.join(dataset_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to load image: {image_file}")
        continue

    # Get corresponding label file
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(labels_path, label_file)

    if not os.path.exists(label_path):
        print(f"Label file not found for image: {image_file}")
        continue

    # Read the label file and draw bounding boxes
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            cls_id, x1, y1, x2, y2 = map(float, line.split())
            
            # Denormalize coordinates
            height, width, _ = image.shape
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the image with bounding boxes

    # Display the image (optional)
    cv2.namedWindow('a',cv2.WINDOW_NORMALqq)
    cv2.imshow('a', image)
    cv2.waitKey(0)

# Close the display window
cv2.destroyAllWindows()
