import cv
import numpy as np
from datetime import datetime
from various_tests import scale_image
from cvutils import *

hc = {}

haars_path = "/Users/soswow/Downloads/OpenCV-2.2.0/data/haarcascades"
def defineHc(key):
    files = {
        "eyes":"/haarcascade_eye.xml",
        "face":"/haarcascade_frontalface_default.xml",
        "mouth":"/Mouth.xml"
    }
    global hc
    if key in files:
        hc[key] = cv.Load(haars_path + files[key])

def find_faces(img):
    return find_part(img, "face")

def find_eyes(img):
    return find_part(img, "eyes", size=(20,20))

def find_mouth(img):
    return find_part(img, "mouth", size=(25,15))

def find_part(img, part, scale=1.05, match=3, size=(25,25)):
    global hc
    if not hc or part not in hc:
        defineHc(part)
    if img.channels > 1:
        gray = cv.CreateImage(sizeOf(img), 8, 1)
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
        img = gray
    return cv.HaarDetectObjects(img, hc[part], cv.CreateMemStorage(0), scale, match, cv.CV_HAAR_DO_CANNY_PRUNING, size)

def main():
    font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1)
    capture = cv.CreateFileCapture("videos/test4.mp4")
    data_path = "/Users/soswow/Downloads/OpenCV-2.2.0/data/haarcascades"
    hc = cv.Load(data_path + "/haarcascade_frontalface_default.xml")
    start = datetime.now()
    frame_n, fps = 0, 0
    cv.DestroyAllWindows()
    while 1:
        frame = scale_image(cv.QueryFrame(capture))
        if not frame:
            break

        dest = cv.fromarray(np.rot90(np.asarray(cv.GetMat(frame))).copy())

        image_size = cv.GetSize(dest)
        grayscale = cv.CreateImage(image_size, 8, 1)
        cv.CvtColor(dest, grayscale, cv.CV_BGR2GRAY)

        faces = cv.HaarDetectObjects(grayscale, hc, cv.CreateMemStorage(0), 1.1, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (25, 25))
        for (x,y,w,h),n in faces:
            cv.Rectangle(dest, (x,y), (x+w,y+h), 255)

        frame_n += 1
        diff_t = (datetime.now() - start).seconds
        if diff_t > 0:
            fps = frame_n/diff_t
            cv.PutText(dest,"%d FPS" % fps, (0,15), font, cv.RGB(255,255,255))

        cv.ShowImage("main", dest)

        key = cv.WaitKey(33)
        if key == 27:
            break


if __name__ == "__main__":
    main()
  
