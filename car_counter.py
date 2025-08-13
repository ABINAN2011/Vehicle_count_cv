import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from sort import *
import math
import csv

def load_coco_class_names():
    return [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

def detect_and_track_vehicles(frame, model, tracker, class_names, conf_threshold, count_line_y, counted_ids, id_positions):
    detections = np.empty((0, 5))
    results = model(frame, stream=True)

    # Detect vehicles of interest
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            if cls < len(class_names):
                label = class_names[cls]
                if label in ["car", "truck", "bus", "motorbike"] and conf >= conf_threshold:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    # Update tracker with detections
    tracked_objects = tracker.update(detections)

    new_counted = 0
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw bounding box, ID, and center point
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cvzone.putTextRect(frame, f'ID:{track_id}', (x1, y1 - 10), scale=1.5, thickness=2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

        # Track vertical positions of tracked IDs for line crossing
        if track_id not in id_positions:
            id_positions[track_id] = []
        id_positions[track_id].append(cy)

        # Check if vehicle crossed the line downward
        if len(id_positions[track_id]) >= 2:
            if id_positions[track_id][-2] < count_line_y <= id_positions[track_id][-1]:
                if track_id not in counted_ids:
                    counted_ids.append(track_id)
                    new_counted += 1
                    # Change line color on crossing
                    cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 255, 0), 5)

    return frame, len(counted_ids)

def generate_count_report(vehicle_counts, fps, frame_skip):
    effective_fps = fps / frame_skip
    total_seconds = int(len(vehicle_counts) / effective_fps)

    print("\n--- Vehicle Count Report (Per Second) ---")
    report = []
    for sec in range(1, total_seconds + 1):
        frame_idx = int(sec * effective_fps) - 1
        if frame_idx < len(vehicle_counts):
            count_at_sec = vehicle_counts[frame_idx]
            report.append((sec, count_at_sec))
            print(f"Second {sec}: {count_at_sec} vehicles counted")

    # Save report as CSV
    with open('vehicle_count_report.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Second", "Vehicle Count"])
        writer.writerows(report)

    print("\nReport saved as 'vehicle_count_report.csv'.")

def vehicle_counter(video_path, save_output_path, model_path="yolov8s.pt", frame_skip=1, conf_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    class_names = load_coco_class_names()
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)

    total_counted_ids = []
    id_positions = {}  # FIXED: dictionary, not list
    vehicle_counts_over_time = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count_line_y = int(height * 0.65)

    out = cv2.VideoWriter(save_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1
        if frame_number % frame_skip != 0:
            continue

        # Draw counting line (red)
        cv2.line(frame, (0, count_line_y), (width, count_line_y), (0, 0, 255), 5)

        # Detect and track vehicles on current frame
        frame, current_count = detect_and_track_vehicles(
            frame, model, tracker, class_names, conf_threshold,
            count_line_y, total_counted_ids, id_positions
        )

        # Display vehicle count on frame
        cvzone.putTextRect(frame, f'Vehicle Count: {current_count}', (50, 50), scale=2, thickness=2)

        vehicle_counts_over_time.append(current_count)

        out.write(frame)

        # Optional: encode frame bytes if streaming e.g. in Streamlit
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = buffer.tobytes()
        yield frame_encoded, vehicle_counts_over_time

    cap.release()
    out.release()

    # Generate vehicle count report per second
    generate_count_report(vehicle_counts_over_time, fps, frame_skip)
