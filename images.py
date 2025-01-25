import argparse
import cv2
from pathlib import Path

from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel
from ultralytics.utils.plotting import Annotator, colors


class ImagesInference:
    """
    Class to run object detection (via SAHI + YOLO) on a folder of images.
    The annotated images will be saved into processed/images by default.
    """

    def __init__(self, model_path, device="cuda", conf_thres=0.25):
        """Initialize the detection model."""
        self.model_path = model_path
        self.device = device
        self.conf_thres = conf_thres
        self.detection_model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model with SAHI interface."""
        self.detection_model = UltralyticsDetectionModel(
            model_path=self.model_path,
            confidence_threshold=self.conf_thres,
            device=self.device
        )

    def run_inference_on_images(self, source):
        """
        Run object detection on a folder of images using SAHI + YOLO.

        Args:
            source (str): Path to the folder containing images.
        """
        # Prepare image output directory
        images_out_dir = Path("processed/images")
        images_out_dir.mkdir(parents=True, exist_ok=True)

        # Collect image paths
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        image_paths = []
        for ext in exts:
            image_paths.extend(Path(source).glob(ext))
        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"No images found in: {source}")
            return

        # Inference and saving
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            annotator = Annotator(img)

            # Perform sliced prediction
            results = get_sliced_prediction(
                img[..., ::-1],
                self.detection_model,
                slice_height=640,
                slice_width=640,
            )

            # Extract detection info and draw bounding boxes
            detection_data = [
                (
                    det.category.name,
                    det.category.id,
                    det.score.value,
                    (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy)
                )
                for det in results.object_prediction_list
            ]

            for det in detection_data:
                cls_name, cls_id, conf, box = det
                label = f"{cls_name} {conf:.2f}"
                annotator.box_label(box, label=label, color=colors(int(cls_id), True))

            annotated_img = annotator.result()

            # Always save annotated image
            save_name = f"{img_path.stem}_annotated{img_path.suffix}"
            cv2.imwrite(str(images_out_dir / save_name), annotated_img)


def parse_args():
    parser = argparse.ArgumentParser(description="SAHI + YOLO Inference on a folder of images (no display).")
    parser.add_argument("--source", type=str, required=True, help="Path to the folder of images.")
    parser.add_argument("--model_path", type=str, default="./model/best.pt",
                        help="Path to your YOLO model weights.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run the model on.")
    return parser.parse_args()


def main():
    args = parse_args()
    images_inference = ImagesInference(
        model_path=args.model_path,
        device=args.device,
    )
    images_inference.run_inference_on_images(source=args.source)


if __name__ == "__main__":
    main()
