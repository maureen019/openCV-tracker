# pip install opencv-python deepface
import cv2
import numpy as np
from RecogniseFace import RecogniseFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
counter = 0
faceRec = RecogniseFace()

while True:
    ret, frame = cap.read()

    if ret: # if frame exists
        frame = faceRec.detectFaces(frame)
        if counter % 40 == 0: # check face every 40 seconds
            try:
                faceRec.checkFace(frame)
            except ValueError as e:
                print(f"Error: {e}")

        counter += 1

        if faceRec.faceMatch:
            cv2.putText(frame, 'MATCH!', (20,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
            cv2.putText(frame, 'Similarity Score: '+str(faceRec.getSimilarityScore())+'%', (210,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
        else:
            cv2.putText(frame, 'NO MATCH!', (20,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
    
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()