import cv
from cvutils import *

@time_took
def get_canny_img(img):
    size = sizeOf(img)
    plate = cv.CreateImage(size, 8, 1)
    cv.Set(plate, 255)
    for k in (50, 100,150,200,250):
#    k=100
        edges = cv.CreateImage(size, 8, 1)
        cv.Canny(img, edges, k-20, k)
#                show_image(edges)
        if k >= 100:
            cv.Dilate(edges,edges)
        else:
            k+=50
#        cv.Erode(edges,edges)
        cv.Set(plate, 255 - k, edges)
    return plate


def main():
#    img = normalize_plane(cv.LoadImage("sample/lena.bmp", iscolor=False))
#    cv.Smooth(img, img, cv.CV_GAUSSIAN, 3, 3)
    cap = cv.CaptureFromCAM(0)
    black = cv.RGB(0, 0, 0)
    while 1:
        img = prepare_bw(cv.QueryFrame(cap))
        plate, time = get_canny_img(img,time_took=True)
        write_info(plate, "%.6f canny" % time,color=black)

        cv.ShowImage("win", plate)
        key = cv.WaitKey(10)
        if key == 27:
            break
#    show_image(plate)

if __name__ == "__main__":
    main()
  