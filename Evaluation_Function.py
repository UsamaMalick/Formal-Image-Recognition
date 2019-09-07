from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv

cascPath = "haarcascadeFiles/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier('haarcascadeFiles/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascadeFiles/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascadeFiles/haarcascade_smile.xml')
model_path = "haarcascadeFiles/gender_detection.model"


def Check_Blurry(image):
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    text = "Not Blurry"
    if fm <= 90: text = "Blurry"
    return text

# def FaceDetection(image):
#     faces = face_cascade.detectMultiScale(image, 1.3, 5)
#     return faces

def FaceDetection(image):
    faces, confidence = cv.detect_face(image)
    count = len(faces)
    return count

def Smile_Eye_Detection(grayScale_image , image):
    faces = face_cascade.detectMultiScale(grayScale_image, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = grayScale_image[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        print("Smiles Found : " , len(smiles))

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (250, 250, 250), 2)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        print("Number of eyes : " , len(eyes))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        return  len(smiles) , len(eyes)
    return 0  , 0

def DetectGender(image):
    face, confidence = cv.detect_face(image)
    classes = ['man','woman']
    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(image[startY:endY,startX:endX])

        # pre-processing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        accuracy = conf[idx] * 100
        return label , accuracy

image = cv2.imread('images/bluryface.jpg')


# load model
model = load_model(model_path)

if image is None:
    print("Could not read input image")
    exit()
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurry = Check_Blurry(grayscale_image)
# face = FaceDetection(grayscale_image)
face = FaceDetection(image)

Gender , accuracy = DetectGender(image)

faceCount = face

# for i in face:
#     faceCount = faceCount + 1

TotalScore = 0

if(blurry == "Not Blurry"):
    TotalScore = 20
else:
    TotalScore = -10

if(faceCount == 1):
    TotalScore = TotalScore + 25
else:
    TotalScore = TotalScore - (10 * faceCount)

Smile , Eyes = Smile_Eye_Detection(grayscale_image , image)
if(Smile >= 1 and Eyes == 2):
    TotalScore = TotalScore + 40
else:
    if(Smile == 1 and Eyes == 1):
        TotalScore = TotalScore + 25

if(Eyes == 2):
    TotalScore = TotalScore + 35

Given_Gender = "man"
if(Gender == Given_Gender and accuracy > 90):
    TotalScore = TotalScore + 20
else:
    TotalScore = TotalScore + 10

if(Gender != Given_Gender):
    TotalScore = TotalScore - 20

print("Picture Quality : " + blurry)
print("Number of Faces : " , faceCount)
print("Gender is : " + Gender)
print("Accuracy of Gender Detection : " , accuracy)


if(TotalScore > 85):
    print("Image is Suitable! " , TotalScore)
else:
    print("Image is not Suitable! " , TotalScore)

cv2.imshow("Evaluation", image)
# press any key to close window
cv2.waitKey()

# release resources
cv2.destroyAllWindows()
