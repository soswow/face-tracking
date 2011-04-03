__author__ = 'soswow'
import cv
from cvutils import *
from skindetect import *

def hsv_test():
    img = cv.LoadImage("sample/img_563.jpg")
    h,s,v= get_hsv_planes(img)

    show_images({"orig": img,
                 "h": h,
                 "s": s,
                 "v": v})

def cam_hsv_test():
    cap = cv.CaptureFromCAM(0)
    while 1:
        img_orig = cv.QueryFrame(cap)
        img = scale_image(img_orig,4)
        h,s,v,inv_s = get_hsv_planes(img)
        y,cr,cb = get_ycrcb_planes(img)

        h_norm = image_empty_clone(h)
        cr_norm = image_empty_clone(cr)
        cb_norm = image_empty_clone(cb)
        cv.Normalize(h,h_norm,0,255,cv.CV_MINMAX)
        cv.Normalize(cr,cr_norm,0,255,cv.CV_MINMAX)
        cv.Normalize(cb,cb_norm,0,255,cv.CV_MINMAX)
        #        r,g,b = split(img)

        show_images({"orig": img,
                     "h": h_norm,
                     "inverted s": inv_s,
                     "v":v,
                     "y":y,
                     "Cr":cr_norm,
                     "Cb":cb_norm,
                     #"R":r,"G":g,"B":b
        })
        key = cv.WaitKey(50)
        if key == 27:
            break


def webcam_hist2d():
    cap = cv.CaptureFromCAM(0)
    cv.NamedWindow("Source", 1)
    cv.NamedWindow("H-S Histogram", 1)
    while 1:
        img = cv.QueryFrame(cap)
        hist_img = get_2d_hist_img(img=img)

        cv.ShowImage("Source", img)
        cv.ShowImage("H-S Histogram", hist_img)

        if cv.WaitKey(20) == 27:
            break

def webcam_hist2d():
    cap = cv.CaptureFromCAM(0)
    cv.NamedWindow("Source", 1)
    cv.NamedWindow("H-S Histogram", 1)
    while 1:
        img = cv.QueryFrame(cap)
        hist_img = get_2d_hist_img(img=img)

        cv.ShowImage("Source", img)
        cv.ShowImage("H-S Histogram", hist_img)

        if cv.WaitKey(20) == 27:
            break

def get_rgb_histogram_images(img, bins=255, width=510):
    r,g,b = get_rgb_planes(img)
    r_hist_img = get_hist_image(get_gray_histogram(r, bins), bins, width)
    g_hist_img = get_hist_image(get_gray_histogram(g, bins), bins, width)
    b_hist_img = get_hist_image(get_gray_histogram(b, bins), bins, width)
    return r_hist_img, g_hist_img, b_hist_img

def webcam_rgb_histograms():
    cap = cv.CaptureFromCAM(0)
    cv.NamedWindow("Source", 1)
    cv.NamedWindow("R Histogram", 1)
    cv.NamedWindow("G Histogram", 1)
    cv.NamedWindow("B Histogram", 1)
    while 1:
        img = cv.QueryFrame(cap)
#        img = scale_image(img, scale_factor=4)
        r,g,b = get_rgb_histogram_images(img)

        cv.ShowImage("Source", img)
        cv.ShowImage("R Histogram", r)
        cv.ShowImage("G Histogram", g)
        cv.ShowImage("B Histogram", b)

        if cv.WaitKey(20) == 27:
            break

def get_source_and_normal_planes(img,aggressive=0.01):
    r,g,b = get_rgb_histogram_images(img,width=255)
    norm_img = normalize(img,aggressive)
    norm_r, norm_g, norm_b =get_rgb_histogram_images(norm_img,width=255)
    return {"img":img,
             "r":r,
             "g":g,
             "b":b,
             "norm":norm_img,
             "norm_r":norm_r,
             "norm_g":norm_g,
             "norm_b":norm_b}

def histogram():
    img = cv.LoadImage("sample/lena.bmp")
    show_images(get_source_and_normal_planes(img))

def webcam_normalize():
    cap = cv.CaptureFromCAM(0)
    while 1:
        img = cv.QueryFrame(cap)
#        dst = image_empty_clone(img)
#        cv.Smooth(img, dst, cv.CV_GAUSSIAN, 7, 7)
        dic = get_source_and_normal_planes(img,aggressive=0.005)
        for name, plane in dic.items():
            cv.ShowImage(name, plane)

        if cv.WaitKey(20) == 27:
            break

def gui():
    cv.NamedWindow("Window")
    def drag(x):
        print x
    cv.CreateTrackbar("track1","Window", 0, 100, drag)
    img = normalize(cv.LoadImage("sample/0003_00000002.jpg"))
    cv.ShowImage("Window",img)
    cv.WaitKey(0)

def test_rgb_planes():
    img = cv.LoadImage("sample/image3114.png")
    r,g,b = get_rgb_planes(img)
    show_images({"img":img, "r":r,"g":g,"b":b})


def test1():
    for src in ["sample/0003_00000002.jpg", "sample/img_563.jpg","sample/lena.bmp"]:
        img = cv.LoadImage(src)
        eq_img = equalize(img)
        norm_img = normalize(img,0.05)
        img_skin = filter_skin(img)
        eq_skin = filter_skin(eq_img)
        norm_skin = filter_skin(norm_img)
        
        show_images({"img":img,"eq":eq_img,"norm":norm_img,
                     "img_skin":img_skin,"eq_skin":eq_skin,"norm_skin":norm_skin})

def test_normalize_plane():
    img = cv.LoadImage("sample/lena.bmp")
    _, g, _ = get_rgb_planes(img)
    g_hist_img = get_hist_image(get_gray_histogram(g, bins=255), bins=255)

    ng = normalize_plane(g)
    ng_hist_img = get_hist_image(get_gray_histogram(ng, bins=255),bins=255)

    show_images({"g":g, "g_hist_img":g_hist_img, "ng":ng,"ng_hist_img":ng_hist_img})


if __name__ == "__main__":
    webcam_normalize()
#    test_normalize_plane()
#    gui()
#    webcam_normalize()
#    histogram()
#    webcam_rgb_histograms()
#    webcam_hist()
#    cam_hsv_test()
#    histogram()
    #main()