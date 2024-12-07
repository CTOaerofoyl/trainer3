from ultralytics import YOLO

model = YOLO('train2/11n-35-trained.pt')

results = model.predict(source=f"rtsp://admin:PAG00319@192.168.1.223:554/live", show=True,device='cuda',stream=True,verbose=False)

for result in results:
    for box in result.boxes:
    # print(result['boxes'])
        print(box.xyxyn.cpu())