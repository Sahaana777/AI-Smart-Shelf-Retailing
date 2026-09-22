from ultralytics import YOLO


def main():
    model = YOLO("models/best.pt")

    results = model.predict(
        source="data/raw/test_shelf.jpg",
        conf=0.5,
        save=True
    )

    for result in results:
        print(result)


if __name__ == "__main__":
    main()