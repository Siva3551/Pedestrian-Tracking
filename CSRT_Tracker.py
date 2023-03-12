
import cv2

# Initialize MIL tracker
tracker = cv2.TrackerCSRT_create()

# Read the video file
video = cv2.VideoCapture('video_1.mp4')

# Get the initial frame
success, frame = video.read()

# Define the region of interest (ROI) for the object to be tracked
bbox = cv2.selectROI(frame, False)

# Initialize the tracker with the first frame and ROI
tracker.init(frame, bbox)

# Loop through the video frames
while True:
    # Read a new frame
    timer = cv2.getTickCount()

    success, frame = video.read()
    if not success:
        break
    
    # Update the tracker with the new frame
    success, bbox = tracker.update(frame)
    
    # Draw the bounding box around the tracked object
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the tracking result
    # FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # print(FPS)

    cv2.imshow('CSRT Tracker', frame)
    
    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close the window
video.release()
cv2.destroyAllWindows()
