from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model
results = model.predict(r"D:\Codes\trainer3\dataset1\images\train\192.168.1.221_bag1_1_0.jpg", save=False, show=True)
print(results[0].boxes)