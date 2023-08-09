from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('YOLOv8n-seg.pt')

    model.train(data='EXAMPLE.yaml', epochs=10, imgsz=640)