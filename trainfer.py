from ultralytics import YOLO
import time
import os

if __name__ == "__main__":
    print("Starting training...")

    start_time = time.time()


    model = YOLO(f'yolo11n.pt')

    # Define a directory to save all trained models
    save_directory = 'train2'
    dataset_path = 'dataset1/data.yaml'

        # Print the training results

    os.makedirs(save_directory, exist_ok=True)

    # Train the model
    model.train(
        data=dataset_path,
        epochs=50,
        imgsz=640,
        batch=-1,  # Set a reasonable batch size
        name=f'11n-35-training', 
        device='cuda',
        save_dir=save_directory,  # Save all models to the same directory
        cache=False,  # Use the pre-cached dataset
        project='run'
    )

    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    model.save(os.path.join(save_directory, f'11n-35-trained.pt'))
