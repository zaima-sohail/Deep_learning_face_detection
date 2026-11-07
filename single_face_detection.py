import argparse
import os
from pathlib import Path
from typing import Any

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from mtcnn import MTCNN


def annotate_faces(
    image: np.ndarray, detector: MTCNN
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    height, width = image.shape[:2]
    for face in faces:
        x, y, box_width, box_height = face["box"]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width - 1, x + box_width)
        y2 = min(height - 1, y + box_height)
        if x1 >= x2 or y1 >= y2:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        confidence = face["confidence"]
        cv2.putText(
            image,
            f"Face: {confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        for point_x, point_y in face["keypoints"].values():
            if 0 <= point_x < width and 0 <= point_y < height:
                cv2.circle(image, (point_x, point_y), 3, (0, 0, 255), -1)

    return image, faces


def detect_faces(image_path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return annotate_faces(image, MTCNN())


def main() -> None:
    default_image = Path(__file__).parent / "images" / "vk.jpg"
    parser = argparse.ArgumentParser(
        description="Detect faces in an image with the MTCNN deep-learning model."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=default_image,
        help=f"Image to analyze (default: {default_image})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the annotated image.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening an image window.",
    )
    args = parser.parse_args()

    image, faces = detect_faces(args.image)
    print(f"Detected {len(faces)} face(s) in {args.image}.")

    if args.output:
        if not cv2.imwrite(str(args.output), image):
            raise OSError(f"Could not save annotated image: {args.output}")
        print(f"Saved annotated image to {args.output}.")

    if not args.no_display:
        cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Face Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
