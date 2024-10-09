import cv2
import time
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation

model = YOLO("yolov8n.pt")
names = model.model.names
cap = cv2.VideoCapture("test2.mp4")  # Replace with your video path

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
line_pts = [(0, 360), (1280, 360)]  # Define your line coordinates

speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts, names=names, view_img=True)

# Number of frames to skip (adjust as necessary)
frame_skip = 1



frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    tracks = model.track(im0, persist=True, show=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)
    

    # Check for key press
    key = cv2.waitKey(1)
    if key == 13:  # Enter key
        break

# Release video capture and writer
cap.release()
video_writer.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
