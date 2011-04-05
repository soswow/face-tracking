import cv
from cvutils import *
from skindetect import *

def get_mask_with_contour(img):
    img = normalize(img, aggressive=0.005)
    mask = skin_mask(img)

    di_mask = image_empty_clone(mask)
    cv.Dilate(mask, di_mask)

    seqs = cv.FindContours(cv.CloneImage(di_mask), cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL)

    c_img = image_empty_clone(mask)
    cv.DrawContours(c_img, seqs, 255, 255, 10, -1)

    er_img = image_empty_clone(c_img)
    cv.Erode(c_img, er_img,iterations=2)

    seqs = cv.FindContours(cv.CloneImage(er_img), cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL)
    er_seq_img = cv.CreateImage(sizeOf(er_img), 8, 3)
    cv.Merge(mask,mask,mask,None,er_seq_img)
    cv.DrawContours(er_seq_img, seqs, cv.RGB(255,50,50), 0, 10, thickness=2)

#    show_image(mask, "mask 1")
#    show_image(di_mask, "dilate mask")
#    show_image(c_img, "contour fill")
#    show_image(er_img,"back erode")
#    show_image(er_seq_img,"back erode contour")

    return er_seq_img

def main():
    cap = cv.CaptureFromCAM(0)
    while 1:
        img = cv.QueryFrame(cap)
        img = normalize(img, aggressive=0.005)
        mask = get_mask_with_contour(img)

        cv.ShowImage("mask", mask)
        key = cv.WaitKey(10)
        if key == 27:
            break

if __name__ == "__main__":
    main()
#    img = cv.LoadImage("sample/img_563.jpg")
#    get_mask_with_contour(img)