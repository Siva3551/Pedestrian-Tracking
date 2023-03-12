import cv2

# tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerKCF_create()
i=0
video = cv2.VideoCapture('video_1.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
success, frame = video.read()

# Initialize the tracker with a bounding box in the first frame
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)

# Update the tracker in subsequent frames
while True:
    timer = cv2.getTickCount()

    success, frame = video.read()
    if not success:
        break
    
    # Update the tracker with the new frame
    success, bbox = tracker.update(frame)
    
    # Draw the bounding box on the frame
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    # FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # print(FPS)


    cv2.imshow("Frame", frame)

    cv2.imwrite('img'+str(i)+'.jpg',frame)
    i+=1

    delay = int(1000 / fps)

    if cv2.waitKey(delay) == 27:
    # if cv2.waitKey(1) & 0xff == ord('q'):
        break
