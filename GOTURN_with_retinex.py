import cv2
import sys
import os
import json

import retinex

with open('config.json', 'r') as f:
    config = json.load(f)

cap = cv2.VideoCapture('video_1.mp4')

videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# tracker= cv2.legacy_TrackerMOSSE.create()
# tracker= cv2.TrackerKCF_create()
# tracker= cv2.TrackerCSRT_create()
tracker= cv2.TrackerGOTURN_create()

#TRACKER INITIALIZATION
success, frame = cap.read()
bbox = cv2.selectROI("Tracking",frame,False)

#frameSize = (frame.shape[0],frame.shape[1])

imge = retinex.SSR(
    frame,59,bbox
    
  )


tracker.init(imge,bbox)
x=bbox


def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

n=0
i=1
img_ms = []
c=[]
while True:

    timer = cv2.getTickCount()
    success, img = cap.read()

    
    msrcr = retinex.SSR(
        img,59,bbox
    )
    # img_msrcr.append(msrcr)
    # n=n+1

    success, bbox = tracker.update(msrcr)
    i=i+1

    if i%5==0:
        tracker.init(img_ms[i-5],c[i-5])

    if success:
        drawBox(msrcr,bbox)
        if bbox[2]>0:
          n=n+1
    else:
        # print(n)
        drawBox(msrcr,x)
        # break
    # tracker.init(msrcr,bbox)

        # cv2.putText(imge, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # success, bbox = tracker.update(img)
        # if success:
        #     drawBox(img_msrcr,bbox)

    # cv2.rectangle(img,(15,15),(200,90),(255,0,255),2)
    # cv2.putText(img, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2);
    # cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);

    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # if fps>60: myColor = (20,230,20)
    # elif fps>20: myColor = (230,20,20)
    # else: myColor = (20,20,230)
    # cv2.putText(img,str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);

    cv2.imshow("Tracking", msrcr)
    # out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (videoWidth,videoHeight))
    # out.write(msrcr)

    # cv2.imwrite('img'+str(i)+'.jpg',msrcr)
    # i+=1

    if cv2.waitKey(1) & 0xff == ord('q'):
       break
    # print(img_ms)