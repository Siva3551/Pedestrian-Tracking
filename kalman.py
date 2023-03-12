import cv2
import numpy as np

# Define the Kalman filter
kalman = cv2.KalmanFilter(4, 2, 0)
state = np.zeros((4, 1), np.float32)


kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.processNoiseCov = np.identity(4, np.float32) * 1e-5
kalman.measurementNoiseCov = np.identity(2, np.float32) * 1e-1
kalman.errorCovPost = np.identity(4, np.float32)


# Set the state transition matrix of the Kalman Filter


# Initialize the object tracker
tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture("video_1.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
dt=1/fps

A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.float32)

# Define the new velocity model
vel = 0.1   # constant acceleration of 0.1 pixels/frame^2
A[0, 2] = dt + vel * dt**2 / 2
A[1, 3] = dt + vel * dt**2 / 2
A[2, 3] = vel * dt


# Read the first frame
ok, frame = video.read()
if not ok:
    print("Cannot read video file")

# Select the object to track
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

state[0] = bbox[0]   # initial x-position
state[1] = bbox[1]   # initial y-position
state[2] = 0   # initial x-velocity
state[3] = 0   # initial y-velocity

kalman.statePost = state

# Loop over the frames and track the object
while True:
    # Read the next frame
    ok, frame = video.read()
    if not ok:
        break

    # Predict the next state using the Kalman filter
    prediction = kalman.predict()

    kalman.transitionMatrix = A

    # Update the tracker with the predicted bounding box
    # bbox = (int(x), int(y), int(w), int(h))
    ok, bbox = tracker.update(frame)

    l = np.array([bbox[0], bbox[1]], np.float32).reshape(2, 1)
    kalman.correct(l)

    # If the tracking is successful, update the Kalman filter with the new measurement
    # if ok:
    #     measurement = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2], np.float32).reshape(2, 1)
    #     # kalman.correct(measurement)

    # Draw the bounding box and the Kalman prediction on the frame
    if ok:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
        # cv2.rectangle(frame, (int(prediction[0] - bbox[2]/2), int(prediction[1] - bbox[3]/2)), (int(prediction[0] + bbox[2]/2), int(prediction[1] + bbox[3]/2)), (0, 0, 255), 2)
        cv2.rectangle(frame, (int(prediction[0]), int(prediction[1])), (int(prediction[0] + bbox[2]), int(prediction[1] + bbox[3])), (0, 0, 255), 2)

    # Display the frame
    delay = int(1000 / fps)

    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(delay)==27 & 0xFF == ord('q'):
        break

# Release the video and close the window
video.release()
cv2.destroyAllWindows()
