import cv
import numpy as np
from datetime import datetime
from various_tests import scale_image

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

        faces = cv.HaarDetectObjects(grayscale, hc, cv.CreateMemStorage(0), 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (50, 50))
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
  
