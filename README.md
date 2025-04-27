# Multiple Vehicle Tracking with Enhanced SORT


---

## Demo
<!-- ![Tracking Example](output/output.gif) -->

## Abstract

This repository contains the implementation of a multiple vehicle tracking system based on enhancements to the **Simple Online and Realtime Tracking (SORT)** algorithm. Key improvements include:
- Experimentation with different **motion models** for better occlusion handling.
- **Reduced computational requirements** by varying observation frequency.
- A tracking system evaluated on real-world datasets showing improved accuracy and runtime efficiency.

---

## Problem Statement

Accurate multiple vehicle tracking is essential for intelligent transportation systems, traffic monitoring, and collision avoidance. This project tackles:
- Reliable **tracking of moving vehicles** from traffic camera footage.
- **Occlusion handling** and **long-duration tracking** using better motion modeling.
- **Efficient computation** for near real-time application.

---

## Methodology

The pipeline follows the enhanced SORT framework, consisting of four major steps:

### Detection
- **Object detection** is assumed to be provided externally or simulated.
- Future integrations could connect this tracker to detectors like **YOLOv3** for full end-to-end processing.

### State Estimation
- Vehicles are modeled with an **extended Kalman Filter**.
- **Three custom motion models** were implemented:
  1. **Immediate difference** between consecutive frames.
  2. **Average velocity** over `n` previous frames.
  3. **Heading-fixed model** assuming near-linear highway motion.

- The Kalman Filter uses bounding box position and velocity components.

### Data Association
- **IoU-based cost matrix** is used to associate detections to tracks.
- The **Hungarian algorithm** solves the assignment problem.
- Association is robust to detection noise and partial occlusions.

### Tracker Management
- **Track creation**: New tracks are initialized when new unmatched detections appear.
- **Track deletion**: Tracks are deleted based on a **Time-To-Live (TTL)** counter if unmatched for several frames, to handle missed detections gracefully.

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jordanbierbrier/multipleObjectTracking.git
   cd multipleObjectTracking
   ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download YOLOv3 configuration and weights:**

    The tracker relies on YOLOv3 for object detection. Download the necessary files:
    - **YOLOv3 configuration file (`yolov3.cfg`)**: [Download here](https://pjreddie.com/darknet/yolo/)
    - **YOLOv3 pre-trained weights (`yolov3.weights`)**: [Download here](https://pjreddie.com/darknet/yolo/) 

    Place both files in the project root directory.

4. **Available Command-Line Arguments**

    You can view the help menu by running:
    ``` 
    python track.py -h
    ```

    This will show:
    ```
    usage: track.py [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH]
                [--freq FREQ] [--discard DISCARD] [--margins MARGINS]
                [--track_length TRACK_LENGTH] [--debug] [--hide_rectangles]

    Vehicle Tracking

    optional arguments:
    -h, --help            show this help message and exit
    --input_path INPUT_PATH
                            Filename of input video. Default 'input_video.mp4'
    --output_path OUTPUT_PATH
                            Filename of output video. Default 'output.mp4'
    --freq FREQ           Frequency of observations. Default 5
    --discard DISCARD     Number of frames without observation before discarding
                            state. Default 40
    --margins MARGINS     Percentage of margins to keep tracking out of frame.
                            Default 0.05
    --track_length TRACK_LENGTH
                            Length of car track drawn to screen. Default 15
    --debug               Add debugging features. Default False
    --hide_rectangles     Hide bounding boxes of cars. Default False
    ```

5. **Sample Run**
    ```
    python track.py --input_path "input.mp4"
    ```

## 📄 Full Report
Find detailed report [here](report.pdf).