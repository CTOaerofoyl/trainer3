from ultralytics import YOLO

model = YOLO('train2/best.pt')

results = model.predict(source=f"rtsp://admin:PAG00319@192.168.1.223:554/live", show=True,device='cuda',iou=0.4,stream=True,verbose=False,conf=0.7)

for result in results:
    for box in result.boxes:
    # print(result['boxes'])
        print(box.xyxyn.cpu())