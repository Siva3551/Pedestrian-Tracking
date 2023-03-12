import cv2

tracker = cv2.TrackerMIL_create()

video = cv2.VideoCapture('video_1.mp4')
success, frame = video.read()

# Initialize the tracker with a bounding box in the first frame
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)

# Update the tracker in subsequent frames
while True:
    success, frame = video.read()
    if not success:
        break
    
    # Update the tracker with the new frame
    success, bbox = tracker.update(frame)
    
    # Draw the bounding box on the frame
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
