from collections import Counter
from ultralytics import YOLO


def main():
    model = YOLO("models/best.pt")

    results = model.predict(
        source="data/raw/test_shelf.jpg",
        conf=0.5
    )

    class_names = model.names

    for result in results:
        detected_classes = result.boxes.cls.tolist()

        counts = Counter(
            class_names[int(class_id)]
            for class_id in detected_classes
        )

        print("\nProduct Count")
        print("-" * 30)

        for product, count in counts.items():
            print(f"{product}: {count}")


if __name__ == "__main__":
    main()