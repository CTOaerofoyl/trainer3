from ultralytics import YOLO
import time
import os

if __name__ == "__main__":
    print("Starting training...")

    start_time = time.time()

    model_name = 'yolo11s.pt'

    model = YOLO(model_name)

    # Define a directory to save all trained models
    save_directory = 'train2'
    dataset ='dataset7'
    dataset_path = f'{dataset}/data.yaml'

        # Print the training results

    os.makedirs(save_directory, exist_ok=True)
    epochs = 50


    # Train the model
    model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=640,
        batch=-1,  # Set a reasonable batch size
        name=f'{model_name}-{epochs}-{dataset}', 
        device='cuda',
        save_dir=save_directory,  # Save all models to the same directory
        cache=False,  # Use the pre-cached dataset
        project='run',
        patience=30,
        freeze=10
    )

    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    model.save(os.path.join(save_directory, f'{model_name}-{epochs}-{dataset}.pt'))
    model.save(os.path.join(save_directory, f'best.pt'))
