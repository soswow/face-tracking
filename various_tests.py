__author__ = 'soswow'
import cv
import sys
import numpy as np

def show_img(img, name="win"):
    cv.ShowImage(name, img)
    while 1:
        key = cv.WaitKey(100)
        if key == 27:
            break


def pgm_test():
    img = cv.LoadImage("/Users/soswow/Documents/Face Detection/att_faces/pgm/s1/1.pgm")
    show_img(img)

def split(src):
    channels = [None] * 4
    size = cv.GetSize(src)
    for i in range(src.channels):
        channels[i] = cv.CreateImage(size, src.depth, 1)
    cv.Split(src,channels[0],channels[1],channels[2],None)
    return channels[:3]

def empty_clone(img,size=None,  channels=None):
    new_img = cv.CreateImage(size or cv.GetSize(img), img.depth, channels or img.channels)
    return new_img

def get_hsv_invs(img):
    new_img = empty_clone(img)
    cv.CvtColor(img, new_img, cv.CV_RGB2HSV)
    h,s,v = split(new_img)
    inv_s = empty_clone(img, channels=1)
    cv.SubRS(s,255,inv_s)
    return h,s,v,inv_s

def get_ycrcb(img):
    new_img = empty_clone(img)
    cv.CvtColor(img, new_img, cv.CV_RGB2YCrCb)
    y,cr,cb = split(new_img)
    return y,cr,cb

def scale_image(img_orig, scale_factor=2):
    orig_size = cv.GetSize(img_orig)
    new_size = (orig_size[0] / scale_factor, orig_size[1] / scale_factor)
    img = empty_clone(img_orig,size=new_size)
    cv.Resize(img_orig, img)
    return img

def show_images(images_dict):
    for name, img in images_dict.items():
        cv.ShowImage(name, img)
    cv.WaitKey(0)

def hsv_test():
    img = cv.LoadImage("sample/img_563.jpg")
    h,s,v,inv_s = get_hsv_invs(img)

    show_images({"orig": img,
                 "h": h,
                 "s": s,
                 "inverted s": inv_s,
                 "v": v})

def get_hist_img(hist, bins, width=500):
    height = 255
    white = cv.RGB(255, 255, 255)
    black = cv.RGB(0, 0, 0)

    img_size = (width, height)
    hist_img = cv.CreateImage(img_size, 8, 1)

    cv.Rectangle(hist_img,
                 (0, 0),
                 img_size,
                 white,cv.CV_FILLED)

    (_, max_value, _, _) = cv.GetMinMaxHistValue(hist)

    scale = width / float(bins)
    x = scale
    for s in range(bins):
        bin_val = cv.QueryHistValue_1D(hist, s)
        y = cv.Round(bin_val * height / max_value)
        cv.Rectangle(hist_img,
                     (x, height -y),
                     (x+scale, height),
                     black,
                     cv.CV_FILLED)
        x+=scale
    return hist_img

def cam_hsv_test():
    cap = cv.CaptureFromCAM(0)
    while 1:
        img_orig = cv.QueryFrame(cap)
        img = scale_image(img_orig,4)
        h,s,v,inv_s = get_hsv_invs(img)
        y,cr,cb = get_ycrcb(img)

        h_norm = empty_clone(h)
        cr_norm = empty_clone(cr)
        cb_norm = empty_clone(cb)
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

    h_plane = empty_clone(hsv, channels=1)
    s_plane = empty_clone(hsv, channels=1)
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


def get_gray_histogram(layer, bins=40):
    hist = cv.CreateHist([bins], cv.CV_HIST_ARRAY, [(0,255)], 1)
    cv.CalcHist([layer], hist)

    return hist


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

def get_rgb_planes(img):
    r = empty_clone(img, channels=1)
    g = empty_clone(img, channels=1)
    b = empty_clone(img, channels=1)
    cv.Split(img, r, g, b, None)
    return r,g,b

def get_rgb_histogram_images(img, bins=255, width=510):
    r,g,b = get_rgb_planes(img)
    r_hist_img = get_hist_img(get_gray_histogram(r, bins), bins, width)
    g_hist_img = get_hist_img(get_gray_histogram(g, bins), bins, width)
    b_hist_img = get_hist_img(get_gray_histogram(b, bins), bins, width)
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
    dst = empty_clone(img)
    rgb = get_rgb_planes(img)
    out_rgb = []
    for plane in rgb:
        equal_plane = empty_clone(plane)
        cv.EqualizeHist(plane, equal_plane)
        out_rgb.append(equal_plane)
    cv.Merge(out_rgb[0],out_rgb[1],out_rgb[2],None,dst)
    return dst

def normalize(img):
    dst = empty_clone(img)
    rgb = get_rgb_planes(img)
    out_rgb = []
    for plane in rgb:
        norm_plane = empty_clone(plane)
#        (_, max_value, _, _) = cv.GetMinMaxHistValue(hist)
#        cv.Threshold(plane, norm_plane,)
        cv.Normalize(plane, norm_plane, 0, 255, cv.CV_MINMAX)
#        hist = get_gray_histogram(plane, 255)
#        hist2 = get_gray_histogram(norm_plane, 255)
#        show_images({'before':get_hist_img(hist, 255),
#                      'after':get_hist_img(hist2, 255)})
        out_rgb.append(norm_plane)
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
    gui()
#    webcam_normalize()
#    histogram()
#    webcam_rgb_histograms()
#    webcam_hist()
#    cam_hsv_test()
#    histogram()
    #main()