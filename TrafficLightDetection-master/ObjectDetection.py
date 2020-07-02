import cv2
import numpy as np

class ObjectMiniDetection:
    def __init__(self):
        self.des = "Object Detection v.1 by DL"
        print(self.des)

    def process(self, frame, min_area=100, max_area=1500, max_object=100, threshold=100, kenel_size=(9, 9), max_size_distance=5):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.ones(kenel_size, np.uint8)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        ret, thresh = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, max_size_distance)
        ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
        watershed = cv2.watershed(frame, markers)

        id = np.array([val for val in range(1, ret + 1)])
        area = np.array([np.sum(markers == val) for val in range(1, ret + 1)])
        select = np.array((area > min_area) & (area < max_area))
        ft = np.array(id[select == True])
        if ft.shape[0] > max_object:
            sli = int((ft.shape[0]-max_object)/2)
            ft = ft[sli:sli+max_object]

        outobj = []
        posobj = []
        for sc in ft:
            aw = watershed == sc
            pos = np.argwhere(aw == True)

            ps1, ps2 = self.bounding_box(pos)

            # color = (0, 255, 0)
            w = ps2[0] - ps1[0] + 1
            h = ps2[1] - ps1[1] + 1
            crop = frame[ps1[1]:ps1[1] + h, ps1[0]:ps1[0] + w]
            outobj.append(crop)
            posobj.append((ps1, ps2))

        return tophat, thresh, dist_transform, outobj, posobj

    def bounding_box(self, points):
        x_coordinates, y_coordinates = zip(*points)
        return (min(y_coordinates), min(x_coordinates)), (max(y_coordinates), max(x_coordinates))



if __name__ == "__main__":
    objd = ObjectMiniDetection()

    # cap = cv2.VideoCapture('demo/video/VID_20200620_174445.mp4')
    #
    # while cap.isOpened():
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     print("FPS: {0}".format(int(fps)))
    #
    #     ret, frame = cap.read()
    #     frame = frame[650:1000, 400:900]
    #     frame = np.array(frame)
    #
    #     _, _, _, _, bound = objd.process(frame, max_object=20)
    #     color = (0, 255, 0)
    #     for b in bound:
    #         cv2.rectangle(frame, b[0], b[1], color)
    #
    #     cv2.imshow('frame', frame)
    #
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
