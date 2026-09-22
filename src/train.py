from ultralytics import YOLO


def main():
    # Load a pretrained YOLO model
    model = YOLO("yolo11n.pt")

    # Train on the custom retail dataset
    model.train(
        data="data/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs",
        name="retail_shelf_model"
    )


if __name__ == "__main__":
    main()