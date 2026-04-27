"""Diagnostic script: probe the trained YOLO model at multiple input sizes."""

import sys
from pathlib import Path

from ultralytics import YOLO
from PIL import Image


def main(image_path: str) -> None:
    img = Image.open(image_path)
    print(f"image size: {img.size}")

    model = YOLO("models/yolov8s_blood.pt")
    print(f"classes: {model.names}")
    print(f"overrides: {getattr(model, 'overrides', None)}")

    for imgsz in (416, 640, 832, 1024, 1280, 1600):
        result = model.predict(image_path, conf=0.05, imgsz=imgsz, verbose=False)[0]
        n = len(result.boxes)
        if n:
            cs = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            counts = {model.names[int(c)]: int((cs == c).sum()) for c in set(cs.tolist())}
            print(
                f"imgsz={imgsz}: {n} boxes, conf [{confs.min():.3f}, {confs.max():.3f}], counts={counts}"
            )
        else:
            print(f"imgsz={imgsz}: 0 boxes")


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_images/Peripheral_blood_smear.jpg"
    if not Path(image).exists():
        print(f"image not found: {image}", file=sys.stderr)
        sys.exit(1)
    main(image)
