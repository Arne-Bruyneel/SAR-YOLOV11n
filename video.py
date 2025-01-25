import argparse
import cv2
from pathlib import Path

from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel
from ultralytics.utils.plotting import Annotator, colors


class VideoInference:
    """
    Class to run object detection (via SAHI + YOLO) on a video.
    Each frame where a 'person' is detected will be saved as a JPG
    in the processed/videos/images folder.
    Optionally, an annotated video can be saved if --save_video is provided.
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

    def run_inference_on_video(
        self,
        source,
        view_img=False,
        save_video=False,
        skip_fps=5,
    ):
        """
        Run sliced inference on a video.

        Args:
            source (str): Path to the video file.
            view_img (bool): Whether to display video frames in a window.
            save_video (bool): Whether to save an annotated video (.avi).
            skip_fps (int): Number of frames per second to process (defaults to 5).
        """
        # Prepare video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {source}")

        # Video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # fallback if FPS is 0 or can't be read

        # We only process skip_fps frames each second
        frame_skip = int(round(fps / skip_fps)) if fps >= skip_fps else 1

        # Prepare output directories
        videos_dir = Path("processed/videos")
        images_dir = videos_dir / "images"
        videos_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Annotated video writer (optional)
        video_writer = None
        if save_video:
            out_path = videos_dir / f"{Path(source).stem}_annotated.avi"
            video_writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                skip_fps,  # we'll output skip_fps frames per second
                (frame_width, frame_height),
            )

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Skip frames to reduce compute
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            annotator = Annotator(frame)
            # Sliced prediction
            results = get_sliced_prediction(
                frame[..., ::-1],
                self.detection_model,
                slice_height=640,
                slice_width=640,
            )

            detection_data = [
                (
                    det.category.name,
                    det.category.id,
                    det.score.value,
                    (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy)
                )
                for det in results.object_prediction_list
            ]

            person_detected = False
            for det in detection_data:
                cls_name, cls_id, conf, box = det
                label = f"{cls_name} {conf:.2f}"
                annotator.box_label(box, label=label, color=colors(int(cls_id), True))

                # Check if it's a person
                if cls_name.lower() == "person":
                    person_detected = True

            # If a person is detected, always save this frame
            if person_detected:
                time_in_seconds = frame_count / fps
                save_name = f"{Path(source).stem}_frame{frame_count}_time{time_in_seconds:.2f}.jpg"
                cv2.imwrite(str(images_dir / save_name), frame)

            # Show or (optionally) save annotated frame
            annotated_frame = annotator.result()

            if view_img:
                cv2.imshow("Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if save_video and video_writer is not None:
                video_writer.write(annotated_frame)

            frame_count += 1

        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="SAHI + YOLO Inference on Video.")
    parser.add_argument("--source", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--model_path", type=str, default="./model/best.pt",
                        help="Path to your YOLO model weights.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run the model on.")
    parser.add_argument("--view_img", action="store_true",
                        help="Display the video in a window (press 'q' to quit).")
    parser.add_argument("--save_video", action="store_true",
                        help="If set, save annotated video as .avi.")
    parser.add_argument("--skip_fps", type=int, default=5,
                        help="Number of frames per second to process (default: 5).")
    return parser.parse_args()


def main():
    args = parse_args()
    video_inference = VideoInference(
        model_path=args.model_path,
        device=args.device,
    )
    video_inference.run_inference_on_video(
        source=args.source,
        view_img=args.view_img,
        save_video=args.save_video,
        skip_fps=args.skip_fps,
    )


if __name__ == "__main__":
    main()
