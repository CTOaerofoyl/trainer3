import os
import glob

def convert_labels(input_path, output_path=None):
    """
    Convert label files from the format `class_id x1_n y1_n x2_n y2_n`
    to `class_id x_center y_center width height`.

    :param input_path: Path to the directory containing label files.
    :param output_path: Path to save the corrected label files. If None, overwrites the originals.
    """
    label_files = glob.glob(os.path.join(input_path, '*.txt'))

    for label_file in label_files:
        with open(label_file, 'r') as file:
            lines = file.readlines()

        corrected_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping malformed line in {label_file}: {line}")
                continue

            class_id, x1, y1, x2, y2 = parts
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

            # Calculate x_center, y_center, width, and height
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            corrected_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Write to the output file
        output_file = label_file if output_path is None else os.path.join(output_path, os.path.basename(label_file))
        with open(output_file, 'w') as file:
            file.writelines(corrected_lines)

        print(f"Processed: {label_file} -> {output_file}")

# Define paths for train and val label directories
train_labels_path = "dataset1/labels/train"
val_labels_path = "dataset1/labels/val"

# Call the function for train and val directories
convert_labels(train_labels_path)
convert_labels(val_labels_path)
