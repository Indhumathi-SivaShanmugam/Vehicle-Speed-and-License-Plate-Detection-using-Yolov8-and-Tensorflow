import cv2
from ultralytics import YOLO
from itertools import zip_longest
from tracker import Tracker

# Load YOLO model
model = YOLO('yolov8s.pt')

# Initialize tracker
tracker = Tracker()

# Line coordinates and offset
red_line_y = 198
blue_line_y = 268
offset = 6

# Video capture
cap = cv2.VideoCapture('test2.mp4')

# Check if video capture is successful
if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

# Initialize speed calculation variables
prev_frame_time = 0
prev_bbox_id = {}
speeds = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model(frame)

    # Process detected objects
    for result in results:
        for detection in result:
            if len(detection) == 6:
                x1, y1, x2, y2, confidence, class_id = detection
            else:
                continue

            # Filter out cars (assuming class_id for cars is 2)
            if class_id == 2:
                # Update tracker
                bbox_id = tracker.update([[x1, y1, x2, y2]])

                # Process tracked objects
                for bbox in bbox_id:
                    x1, y1, x2, y2, id = bbox
                    cy = (y1 + y2) // 2

                    # Calculate speed
                    curr_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if id in prev_bbox_id:
                        time_diff = (curr_frame_time - prev_bbox_id[id]['time']) / 1000.0  # Convert to seconds
                        distance = ((x1 + x2) // 2 - prev_bbox_id[id]['pos'][0]) ** 2 + ((y1 + y2) // 2 - prev_bbox_id[id]['pos'][1]) ** 2
                        speed = (distance ** 0.5) / time_diff  # Speed in pixels/ms
                        speeds[id] = speed * 3.6  # Convert to km/h (assuming 1 pixel/ms = 3.6 km/h)
                    else:
                        speeds[id] = 0

                    # Update previous position and time
                    prev_bbox_id[id] = {'pos': [(x1 + x2) // 2, (y1 + y2) // 2], 'time': curr_frame_time}

                    # Display speed when vehicle crosses the blue line
                    if blue_line_y - offset < cy < blue_line_y + offset:
                        cv2.putText(frame, 'Speed: {:.2f} km/h'.format(speeds[id]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw lines
    cv2.line(frame, (172, red_line_y), (774, red_line_y), (0, 0, 255), 2)
    cv2.line(frame, (8, blue_line_y), (927, blue_line_y), (255, 0, 0), 2)

    # Save frame
    frame_filename = 'detected_frames/frame_{}.jpg'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    cv2.imwrite(frame_filename, frame)

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()




