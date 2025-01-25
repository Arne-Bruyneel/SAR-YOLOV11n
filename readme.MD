# SAHI + YOLO Inference on Videos and Images

This repository contains two separate scripts for running object detection using [SAHI](https://github.com/obss/sahi) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics):

- **`video.py`** for inferring on video files  
- **`images.py`** for inferring on a folder of images  

The project automatically performs **sliced inference** using SAHI, meaning large images (or frames) are broken into slices for more accurate detection. We specifically showcase how to detect persons and save frames containing persons during video inference, but the setup is general for any class label(s).


## Features

1. **Video Inference** (`video.py`):
   - Processes a video, optionally skipping frames for faster computation.
   - Saves an annotated video file (`.avi`) if desired.
   - Always saves individual frames (as `.jpg`) when a person is detected, including the time stamp in the video.

2. **Image Folder Inference** (`images.py`):
   - Processes all images in a folder (supports common image extensions).
   - Optionally displays and/or saves annotated images.

3. **Configurable Command-Line Arguments**:
   - `--source`: Path to the video file or image folder.
   - `--model_path`: Path to the YOLO model weights (`.pt`).
   - `--device`: Choose between CPU (`cpu`) or GPU (`cuda`).
   - `--view_img`: View the results in a pop-up window.
   - `--save_video` (video script): Save the annotated video (`.avi`).
   - Additional optional parameters (e.g., `--skip_fps` in the video script).

4. **Organized Outputs**:
   - **Video** outputs are saved under `processed/videos/`
     - The annotated video (if `--save_video` is used) goes here.
     - Frames with detected persons are saved in `processed/videos/images/`.
   - **Image** outputs are saved under `processed/images/` as annotated images.

---

## Datasets & Training

We use two primary datasets for training:

- **LADD (LACMUS Drone Dataset)**  
  - [Official page](https://datasetninja.com/lacmus-drone-dataset)  
  - Drone imagery designed for search-and-rescue, featuring diverse scenes.

- **HERIDAL**  
  - [Official page](http://ipsar.fesb.unist.hr/HERIDAL%20database.html)  
  - High-resolution UAV imagery with human presence, making it suitable for person detection tasks.

### Image Preparation

- **Original Image Size**: 4000×3000 pixels  
- **Tiling**: Each original image is split into 640×640 patches (tiles), with padding added as necessary when the dimensions didn’t divide evenly.  
- **Result**: ~105,000 tiles for training.

### Hardware

- **GPU**: NVIDIA A40 with 48 GB VRAM  
- This large memory capacity facilitated training on high-resolution image tiles using sliced inference.

---

## Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/Arne-Bruyneel/SAR-YOLOV11n.git
cd SAR-YOLOV11n
```

### 2. Install Required Packages

Use a virtual environment (optional but recommended) or install globally:

```bash
pip install ultralytics
pip install sahi
```

Or, if you prefer, install requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Model Weights

Place your YOLO model weights (e.g., `best.pt`) in a convenient directory, such as `./model/best.pt`.

---

## Usage

### A. Video Inference

```bash
python video.py \
    --source ./data/video/{your-video}.mp4 \
    --model_path ./model/best.pt \
    --device cuda \
    --view_img \
    --save_video \
    --skip_fps 5
```

- **Arguments**:
  - `--source`: The path to your video file.
  - `--model_path`: Path to your YOLO `.pt` weights.
  - `--device`: Choose `cpu` or `cuda` (if you have a GPU).
  - `--view_img`: Displays the annotated frames in a pop-up window (press `q` to quit).
  - `--save_video`: Saves the annotated video in `processed/videos/`.
  - `--skip_fps`: Number of frames per second to **process** (defaults to `5`).  
    *(For instance, if the video is 30 FPS, this will skip every 6 frames, effectively processing 5 frames each second.)*

**Outputs**:
- If a person is detected in a processed frame, the entire frame is saved as a JPG in `processed/videos/images/`.
- If `--save_video` is used, an annotated video (`.avi`) is also saved in `processed/videos/`.

#### Frame Naming Convention

When a frame is saved, it follows the naming pattern:

```
<video_name>_frame<frame_index>_time<time_in_seconds>.jpg
```

Example: `cut-flyover_frame1044_time34.80.jpg`

- **`cut-flyover`**: The original video filename (without extension).
- **`frame1044`**: The sequential frame number (based on the total frames read in the video).
- **`time34.80`**: The approximate time into the video when this frame was processed (in seconds).

This helps you locate the exact moment in the video where a detection occurred. You can modify this logic in `video.py` if you prefer another naming scheme.

---

### B. Image Folder Inference

```bash
python images.py \
    --source ./data/images \
    --model_path ./model/best.pt \
    --device cuda \
    --view_img \
    --save_img
```

- **Arguments**:
  - `--source`: A folder containing images (supports `.jpg, .jpeg, .png, .bmp, .tif, .tiff`).
  - `--model_path`: Path to your YOLO `.pt` weights.
  - `--device`: Choose `cpu` or `cuda`.
  - `--view_img`: Displays annotated images one by one (press a key to advance, `q` to quit).
  - `--save_img`: Saves annotated copies of the images to `processed/images/`.

**Outputs**:
- Annotated images in `processed/images/` (if `--save_img` is used).

---

## Notes and Tips

- The **person detection** logic is hard-coded by class name (`cls_name.lower() == "person"`). If your YOLO model uses a different name or ID for persons, update accordingly in `video.py`.
- You can adjust **slice dimensions** (`slice_height`, `slice_width`) for your use case.
- If you need to speed up or slow down the processing of videos, adjust `--skip_fps`.
- By default, only frames containing `"person"` are saved from videos. Remove or edit this logic if you want to save all frames or detect a different class.

---

