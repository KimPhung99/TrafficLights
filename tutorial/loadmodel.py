import Model
import cv2

model = Model.Model(modelSaveName="model.h5")
model.loadding()

img = cv2.imread("../demo/img/IMG_1160.JPG")
# print(img)
print(model.predict([img]))