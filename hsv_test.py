import cv

from cvutils import *
from skindetect import clear_hist_in_range, h2cv_values

def main():
    img = cv.LoadImage("sample/image3114.png")
    h,s,v = get_hsv_planes(img)

    hist = get_gray_histogram(h,180,180)
    clear_hist_in_range(hist, *h2cv_values(50,340))

    hist_img = get_hist_image(hist,180,500)
    h_mask = image_empty_clone(h)
    cv.CalcBackProject((h,), h_mask, hist)

    show_images({"img":img, "h":h, "h_mask":h_mask, "hist_img":hist_img})


if __name__ == "__main__":
    main()