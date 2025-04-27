import cv2
import numpy as np
import argparse
import time
import json
from detect import detect_cars_single_image
from filter import KalmanFilter1
from assignment import assign


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    freq = args.freq
    debug = args.debug
    discard_time = args.discard
    margins = args.margins
    track_length = args.track_length
    hide_rectangles = args.hide_rectangles

    cap = cv2.VideoCapture(input_path)

    # For evaluation purposes
    tracked = {"pt": [], "id": [], "frame_num": []}

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_num = 0
    num_of_cars = 0
    total_filters = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        detections = []

        # Object detection
        if frame_num % freq == 0:
            detections, confidences = detect_cars_single_image(frame)

        n_total_filters = len(total_filters)
        n_detections = len(detections)

        if detections:  # If there are detections/observations available
            for detection in detections:
                x, y, w, h = detection
                if not hide_rectangles:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            filter_states = [total_filters[i].get_state() for i in range(n_total_filters)]

            if not total_filters:  # No filters tracking
                for i, detection in enumerate(detections):
                    new_kf = KalmanFilter1(
                        detection, confidences[i], num_of_cars + 1, freq, discard_time, track_length
                    )
                    num_of_cars += 1
                    total_filters.append(new_kf)

                if debug:
                    add_debug_info(frame, n_detections, frame_num, fps, "D + No Filt")

            elif n_total_filters == n_detections:
                _, det_indx, cost = assign(filter_states, detections)
                temp_filters = []

                for i, filter in enumerate(total_filters):
                    if cost[i] >= 0.99:  # New detection
                        new_kf = KalmanFilter1(
                            detections[det_indx[i]], confidences[det_indx[i]], num_of_cars + 1, freq, discard_time, track_length
                        )
                        num_of_cars += 1
                        temp_filters.append(new_kf)
                    else:
                        filter.update(detections[det_indx[i]], confidences[det_indx[i]])

                total_filters += temp_filters

                if debug:
                    add_debug_info(frame, n_detections, frame_num, fps, "Filt=Det")

            elif n_total_filters < n_detections:  # More detections than filters
                _, det_indx, _ = assign(filter_states, detections)

                for i, filter in enumerate(total_filters):
                    filter.update(detections[det_indx[i]], confidences[det_indx[i]])

                for i in range(n_detections - n_total_filters):
                    j = n_detections - i - 1
                    new_kf = KalmanFilter1(
                        detections[det_indx[j]], confidences[det_indx[j]], num_of_cars + 1, freq, discard_time, track_length
                    )
                    num_of_cars += 1
                    total_filters.append(new_kf)

                if debug:
                    add_debug_info(frame, n_detections, frame_num, fps, "Det > Filt")

            elif n_total_filters > n_detections:  # More filters than detections
                _, det_indx, cost = assign(filter_states, detections)
                temp_filters = []

                for i, filter in enumerate(total_filters):
                    if cost[i] >= 0.99:
                        new_kf = KalmanFilter1(
                            detections[det_indx[i]], confidences[det_indx[i]], num_of_cars + 1, freq, discard_time, track_length
                        )
                        num_of_cars += 1
                        temp_filters.append(new_kf)
                    elif det_indx[i] < n_detections:
                        filter.update(detections[det_indx[i]], confidences[det_indx[i]])

                total_filters += temp_filters

                if debug:
                    add_debug_info(frame, n_detections, frame_num, fps, "Det < Filt")

            total_filters = update_filters(
                total_filters, frame, width, height, margins, track_length, hide_rectangles, tracked, frame_num
            )

        else:  # No detections available
            if total_filters:  # There are filters for prediction
                total_filters = update_filters(
                    total_filters, frame, width, height, margins, track_length, hide_rectangles, tracked, frame_num
                )

                if debug:
                    add_debug_info(frame, n_detections, frame_num, fps, "No D. Filt")

        # Display the resulting frame
        frame_num += 1
        percentage_complete = (frame_num / frame_count) * 100
        print(f"{percentage_complete:.2f}%")

        out_vid.write(frame)

    cap.release()
    cv2.destroyAllWindows()

    if debug:
        with open("metadata.json", "w") as outfile:
            json.dump(tracked, outfile)


def update_filters(total_filters, frame, width, height, margins, track_length, hide_rectangles, tracked, frame_num):
    """
    Update the state of all filters, handle predictions, and manage their visualization.

    Args:
        total_filters (list): List of Kalman filters tracking objects.
        frame (numpy.ndarray): Current video frame.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
        margins (float): Percentage of margins to keep tracking out of frame.
        track_length (int): Length of the track to be drawn.
        hide_rectangles (bool): Whether to hide bounding boxes.
        tracked (dict): Dictionary to store tracking information.
        frame_num (int): Current frame number.

    Returns:
        list: Updated list of filters to keep.
    """
    to_keep = []

    for filter in total_filters:
        filter.predict()

        tracked["id"].append(filter.id)
        tracked["pt"].append([float(filter.x[0] + (filter.x[2] * 0.5)), float(filter.x[1] + (filter.x[3] * 0.5))])
        tracked["frame_num"].append(frame_num)

        x, y, w, h = filter.get_state()
        if filter.lost():
            to_keep.append(False)
        elif filter.in_bounds(width, height, margins):
            x1_pred = int(x)
            y1_pred = int(y)
            x2_pred = int(x + w)
            y2_pred = int(y + h)
            if not hide_rectangles:
                cv2.rectangle(frame, (x1_pred, y1_pred), (x2_pred, y2_pred), filter.colour, 1)
                cv2.putText(
                    frame,
                    f"ID: {filter.id}",
                    (x1_pred - 5, y1_pred - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    filter.colour,
                    1,
                    cv2.LINE_AA,
                )
            to_keep.append(True)

            if track_length:
                for i in range(1, len(filter.pts)):
                    if filter.pts[i - 1] is None or filter.pts[i] is None:
                        continue
                    thickness = int(np.sqrt(track_length / float(i + 1)) * 2.5)
                    cv2.line(frame, filter.pts[i - 1], filter.pts[i], filter.colour, thickness)
        else:
            to_keep.append(False)

    return [element for element, keep in zip(total_filters, to_keep) if keep]


def add_debug_info(frame, n_detections, frame_num, fps, status):
    """
    Adds debug information as overlay text on a video frame.

    Args:
        frame (numpy.ndarray): The video frame on which the debug information will be drawn.
        n_detections (int): The number of detections in the current frame.
        frame_num (int): The current frame number in the video sequence.
        fps (float): The frames per second (FPS) of the video processing.
        status (str): A status message to display on the frame.

    Returns:
        None: The function modifies the input frame in place.
    """
    cv2.putText(frame, f"Detections: {n_detections}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, status, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame Num: {frame_num}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Vehicle Tracking")
    parser.add_argument("--input_path", help="Filename of input video. Default 'input_video.mp4'", type=str, default="input_video.mp4")
    parser.add_argument("--output_path", help="Filename of output video. Default 'output.mp4'", type=str, default="output.mp4")
    parser.add_argument("--freq", help="Frequency of observations. Default 5", type=int, default=5)
    parser.add_argument("--discard", help="Number of frames without observation before discarding state. Default 40", type=int, default=40)
    parser.add_argument("--margins", help="Percentage of margins to keep tracking out of frame. Default 0.05", type=float, default=0.05)
    parser.add_argument("--track_length", help="Length of car track drawn to screen. Default 15", type=int, default=15)
    parser.add_argument("--debug", help="Add debugging features. Default False", action="store_true")
    parser.add_argument("--hide_rectangles", help="Hide bounding boxes of cars. Default False", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    main(args)
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
