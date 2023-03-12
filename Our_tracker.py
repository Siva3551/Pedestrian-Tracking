import cv2
import retinex
import json
import numpy as np
from numpy.linalg import inv


kalman = cv2.KalmanFilter(4, 2, 0)
state = np.zeros((4, 1), np.float32)


kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.processNoiseCov = np.identity(4, np.float32) * 1e-5
kalman.measurementNoiseCov = np.identity(2, np.float32) * 1e-1
kalman.errorCovPost = np.identity(4, np.float32)

# Load the template and the video
with open('config.json', 'r') as f:
    config = json.load(f)

cap = cv2.VideoCapture('video_1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
i=0

videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

#TRACKER INITIALIZATION
success, frame = cap.read()
x = cv2.selectROI("Tracking",frame,False)

state[0] = x[0]   
state[1] = x[1]   
state[2] = 0   
state[3] = 0  

kalman.statePost = state

# template = frame[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]


template = retinex.template(
    frame,39,x
  )

# Define the method of template matching
method = cv2.TM_CCOEFF_NORMED
prediction = kalman.predict()

while True:
    # Read the frame from the video
    timer = cv2.getTickCount()
    ret, frame = cap.read()

    frame = retinex.SSR(
        frame,39,x
    )

    crop = frame[int((x[1]-10)):int((x[1]+x[3]+10)),int((x[0]-20)):int((x[0]+x[2]+20))]

    kalman.transitionMatrix = A

    if ret:
        res = cv2.matchTemplate(crop, template, method)

        # Get the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


        top_left = (max_loc[0] + int((x[0]-20)), max_loc[1]+int((x[1]-10)))
        bottom_right = (top_left[0] + x[2], top_left[1] + x[3])

        m = (top_left[0],top_left[1],x[2],x[3])
        l = np.array([m[0], m[1]], np.float32).reshape(2, 1)
      
        residual = l - (kalman.measurementMatrix @ prediction)

        c=np.linalg.norm(residual)

        if c>40:
            # if k<0.5:
            x=(int(prediction[0]),int(prediction[1]),x[2],x[3])
          
        else :
            x=m
            kalman.correct(l)
            if max_val>0.2:
                prediction = kalman.predict()


        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)


        cv2.imshow('Object Tracking', frame)

        # cv2.imwrite('img'+str(i)+'.jpg',frame)
        # i+=1


        delay = int(1000 / fps)
        if cv2.waitKey(delay)==27 & 0xFF == ord('q'):
            break
    else:
        break

    # Release the video and close all windows
cap.release()
cv2.destroyAllWindows()
