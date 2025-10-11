# YOLO Labeling Tool

![image_thumbnail](https://github.com/jonathanrandall/yolo_labelling_tool/blob/main/thumbnail.png)

A GUI-based annotation tool for creating YOLO format datasets with support for bounding boxes and keypoints (pose estimation).

## Features

- Draw and edit bounding boxes
- Add and manage keypoints with visibility flags
- YOLO model inference integration
- Support for multiple classes
- Auto-save with unique key numbering
- Export in YOLO format (compatible with YOLOv8/v11)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Note: `tkinter` is required but typically comes pre-installed with Python. If you encounter tkinter-related errors, install it via your system package manager:

- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **macOS**: Usually included with Python
- **Windows**: Usually included with Python

## Launch the Tool

```bash
python label_tool.py
```

Or from the project directory:

```bash
python yolo_gui/label_tool.py
```

## Quick Start Guide

### 1. Initial Setup

1. **Load Image Directory**: Click "Load Image Directory" and select a folder containing your images (supports .jpg, .jpeg, .png, .bmp)
2. **Set Output Directory**: Click "Set Output Directory" to choose where labeled data will be saved
3. **Configure Classes**: Set the "Box/Keypoint Class" ID and "Num Keypoint Classes" (if using keypoint mode)

### 2. Annotation Modes

The tool has two annotation modes:

#### Box Mode
Draw and edit bounding boxes for object detection.

**Workflow:**
- Drag on the canvas to draw a new box
- Click on a box to select it (shows resize handles)
- Drag the handles to resize the box
- Press "Delete Selected Box" to remove a box

#### Keypoint Mode
Draw boxes and add keypoints for pose estimation.

**Workflow:**
1. Drag to draw a new box (auto-selected after creation)
2. Click inside the selected box to add keypoints **in order** (0, 1, 2, ..., n-1)
3. Click on an existing keypoint to toggle its visibility (visible/invisible)
4. Click on another box to select it
5. Use "Clear Box Keypoints" to remove all keypoints from the selected box

**Keypoint Order**: Add keypoints in sequential order starting from class 0. For example, for a person with 3 keypoints:
- Class 0: Head
- Class 1: Left shoulder
- Class 2: Right shoulder

### 3. Using YOLO Model Inference (Optional)

1. Click "Load Model" and select a YOLO .pt model file
2. Configure inference settings with "Inference Settings" (confidence, IOU thresholds)
3. Click "Run Inference" to automatically detect and annotate objects
4. Edit the detected annotations as needed

### 4. Saving Annotations

Click "Save & Next" to:
- Save the current image and annotations
- Automatically move to the next image
- Generate unique filenames with key counters

## Output Format

The tool creates the following directory structure:

```
output_directory/
├── images/
│   ├── image1_1.jpg
│   ├── image2_2.jpg
│   └── ...
└── labels/
    ├── image1_1.txt
    ├── image2_2.txt
    └── ...
```

### Label Format

**Box-only annotations:**
```
class_id x_center y_center width height
```

**Box with keypoints (YOLO-Pose format):**
```
class_id x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
```

Where:
- All coordinates are normalized (0-1)
- `x_center`, `y_center`: Box center coordinates
- `width`, `height`: Box dimensions
- `kpN_x`, `kpN_y`: Keypoint N coordinates
- `kpN_v`: Keypoint N visibility (0=not labeled, 1=labeled and visible, 2=labeled but occluded)

Missing keypoints are represented as `0.0 0.0 0`.

## Keyboard Shortcuts

(Currently all actions use mouse/buttons - keyboard shortcuts can be added)

## Tips and Best Practices

1. **Set Num Keypoint Classes First**: Before annotating in keypoint mode, set the number of keypoint classes in the settings
2. **Annotation Order**: In keypoint mode, always add keypoints in order (0, 1, 2, ..., n-1) for consistency
3. **Use Inference**: If you have a pre-trained model, use inference as a starting point and correct as needed
4. **Regular Saves**: Use "Save & Next" frequently to avoid losing work
5. **Check Status Bar**: The status bar at the bottom shows helpful messages about your current action

## Class Configuration

### Box/Keypoint Class
The class ID for the bounding box (e.g., 0 for person, 1 for car, etc.)

### Num Keypoint Classes
The total number of keypoint types in your dataset. For example:
- COCO human pose: 17 keypoints
- Custom 3-point: 3 keypoints
- Face landmarks: 5, 68, or more keypoints

This number must be consistent across all annotations in your dataset.

## Troubleshooting

### Images don't appear
- Check that image files have supported extensions (.jpg, .jpeg, .png, .bmp)
- Ensure the canvas has loaded (try resizing the window)

### Can't save annotations
- Make sure both image directory and output directory are set
- Check that you have write permissions to the output directory

### YOLO inference not working
- Verify the model file is a valid YOLO .pt file
- Check that the model is compatible with the ultralytics library
- Ensure the model type matches your annotation needs (detection vs. pose)

## File Structure

```
yolo_gui/
├── label_tool.py          # Main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── key_counter.json      # Auto-generated key counter (do not edit)
```

## Technical Details

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

### Dependencies
- **Pillow**: Image loading and manipulation
- **ultralytics**: YOLO model integration
- **tkinter**: GUI framework (standard library)

## Contributing

Feel free to submit issues or pull requests for improvements.

## License

MIT License (or specify your license)


