import ObjectDetection as od
import Model
import cv2
import numpy as np
import time

#
model = Model.Model(modelSaveName="model.h5", input_size=(70, 70))
model.loadding()

objd = od.ObjectMiniDetection()

cap = cv2.VideoCapture('../demo/video/VID_20200620_174445.mp4')
print(cap.isOpened())
totalFrame = 0
start = time.time()

while cap.isOpened():
    totalFrame += 1
    ret, frame = cap.read()
    frame = frame[650:1000, 400:900]
    frame = np.array(frame)

    _, _, _, objs, bound = objd.process(frame, max_object=20)
    if len(objs) > 0:
        predict = model.predict(objs)
        cls = np.argmax(predict, axis=1)
        axm = np.amax(predict, axis=1)
        # print("Cla: ", cls)
        # print("Axm: ", axm)

        for i in range(len(bound)):
            if axm[i] > 0.9:
                if cls[i] == 2:
                    color = (0, 255, 0)
                    label = "Green Light"
                elif cls[i] == 1:
                    color = (255, 0, 0)
                    label = "Yellow Light"
                else:
                    label = "Red Light"
                    color = (0, 0, 255)

                cv2.rectangle(frame, bound[i][0], bound[i][1], color, 2)

                labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
                _x1 = bound[i][0][0]
                _y1 = bound[i][0][1]  # +int(labelSize[0][1]/2)
                _x2 = _x1 + labelSize[0][0]
                _y2 = bound[i][0][1] - int(labelSize[0][1])

                cv2.rectangle(frame, bound[i][0], bound[i][1], color)
                cv2.rectangle(frame, (_x1, _y1), (_x2, _y2), color, cv2.FILLED)
                cv2.putText(frame, label, (bound[i][0][0], bound[i][0][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    end = time.time()
    fps = int(totalFrame / (end - start))
    print("Estimated frames per second : {0}".format(fps))

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
