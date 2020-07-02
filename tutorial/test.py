import ObjectDetection as od
import Model
import cv2
import numpy as np


model = Model.Model()
model.loadding()

objd = od.ObjectMiniDetection()

cap = cv2.VideoCapture('../demo/video/VID_20200620_174445.mp4')
print(cap.isOpened())
while cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS: {0}".format(int(fps)))

    ret, frame = cap.read()
    frame = frame[650:1000, 400:900]
    frame = np.array(frame)

    _, _, _, objs, bound = objd.process(frame, max_object=20)
    predict = model.predict(np.array(objs))
    print("Predict: ", predict)
    color = (0, 255, 0)
    for b in bound:
        cv2.rectangle(frame, b[0], b[1], color)

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
