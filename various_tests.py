__author__ = 'soswow'
import cv
from cvutils import *

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

def get_hist_2d_img(img):
    size = cv.GetSize(img)
    hsv = cv.CreateImage(size, 8, 3)
    cv.CvtColor(img, hsv, cv.CV_BGR2HSV)

    # Extract the H and S planes

    h_plane = image_empty_clone(hsv, channels=1)
    s_plane = image_empty_clone(hsv, channels=1)
    cv.Split(hsv, h_plane, s_plane, None, None)
    planes = [h_plane, s_plane]

    h_bins = 30
    s_bins = 32
    #    hist_size = [h_bins, s_bins]

    # hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
    h_ranges = [0, 180]
    # saturation varies from 0 (black-gray-white) to
    # 255 (pure spectrum color)
    s_ranges = [0, 255]
    ranges = [h_ranges, s_ranges]

    hist = cv.CreateHist([h_bins, s_bins], cv.CV_HIST_ARRAY, ranges, 1)
    cv.CalcHist(planes, hist)
    (_, max_value, _, _) = cv.GetMinMaxHistValue(hist)

    scale = 10
    hist_img = cv.CreateImage((h_bins*scale, s_bins*scale), 8, 3)

    for h in range(h_bins):
        for s in range(s_bins):
            bin_val = cv.QueryHistValue_2D(hist, h, s)
            intensity = cv.Round(bin_val * 255 / max_value)
            cv.Rectangle(hist_img,
                         (h*scale, s*scale),
                         ((h+1)*scale - 1, (s+1)*scale - 1),
                         cv.RGB(intensity, intensity, intensity),
                         cv.CV_FILLED)
    return hist_img

def webcam_hist2d():
    cap = cv.CaptureFromCAM(0)
    cv.NamedWindow("Source", 1)
    cv.NamedWindow("H-S Histogram", 1)
    while 1:
        img = cv.QueryFrame(cap)
        hist_img = get_hist_2d_img(img)

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
        hist_img = get_hist_2d_img(img)

        cv.ShowImage("Source", img)
        cv.ShowImage("H-S Histogram", hist_img)

        if cv.WaitKey(20) == 27:
            break

def get_rgb_histogram_images(img, bins=255, width=510):
    r,g,b = get_three_planes(img)
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


def histogram():
    img = cv.LoadImage("sample/0003_00000002.jpg")
    r,g,b = get_rgb_histogram_images(img,width=255)
    norm_img = normalize(img)
    norm_r, norm_g, norm_b =get_rgb_histogram_images(norm_img,width=255)
    show_images({"img":img,
                 "r":r,
                 "g":g,
                 "b":b,
                 "norm":norm_img,
                 "norm_r":norm_r,
                 "norm_g":norm_g,
                 "norm_b":norm_b})

def webcam_normalize():
    cap = cv.CaptureFromCAM(0)
    cv.NamedWindow("Source", 1)
    cv.NamedWindow("Normalized", 1)
    while 1:
        img = cv.QueryFrame(cap)
#        img = scale_image(img, scale_factor=4)
#        eq_img = equalize(img)
        norm_img = normalize(img)

        cv.ShowImage("Source", img)
        cv.ShowImage("Normalized", norm_img)

        if cv.WaitKey(20) == 27:
            break

def equalize(img):
    dst = image_empty_clone(img)
    rgb = get_rgb_planes(img)
    out_rgb = []
    for plane in rgb:
        equal_plane = image_empty_clone(plane)
        cv.EqualizeHist(plane, equal_plane)
        out_rgb.append(equal_plane)
    cv.Merge(out_rgb[0],out_rgb[1],out_rgb[2],None,dst)
    return dst

def gui():
    cv.NamedWindow("Window")
    def drag(x):
        print x
    cv.CreateTrackbar("track1","Window", 0, 100, drag)
    img = normalize(cv.LoadImage("sample/0003_00000002.jpg"))
    cv.ShowImage("Window",img)
    cv.WaitKey(0)

if __name__ == "__main__":
    pass
#    gui()
#    webcam_normalize()
#    histogram()
#    webcam_rgb_histograms()
#    webcam_hist()
#    cam_hsv_test()
#    histogram()
    #main()