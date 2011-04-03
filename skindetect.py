import cv
import numpy as np

from cvutils import *

def _hsv2cv_values(func, max, min=0, *hsv_values):
    result = []
    for val in hsv_values:
        if min > val > max:
            raise Exception("Value not in range")
        result.append(int(func(val)))
    return result

def h2cv_values(*hh):
    return _hsv2cv_values(lambda h:h/2, 360, 0, *hh)

def probability_to_255(*sv):
    return _hsv2cv_values(lambda sv:sv*255, 255, 0, *sv)

def clear_hist_in_range(hist, start, stop):
    for k in range(start, stop):
        hist.bins[k] = 0

def get_filtered_plane(plane, filter_ranges, bins, ranges_func):
    hist = get_gray_histogram(plane, bins=bins)
    for start, stop in filter_ranges:
        start, stop =  ranges_func(start, stop) #0.3, 1.0
        clear_hist_in_range(hist, start, stop)
    mask = image_empty_clone(plane)
    cv.CalcBackProject((plane,), mask, hist)
    return mask

def filter_by_hsv(img, ranges):
    h,s,v = get_hsv_planes(img)
    
    h_mask = get_filtered_plane(h, ranges["h"], 180, h2cv_values)
    s_mask = get_filtered_plane(s, ranges["s"], 255, probability_to_255)
    v_mask = get_filtered_plane(v, ranges["v"], 255, probability_to_255)

    hsv_mask = image_empty_clone(v_mask)
    cv.And(v_mask, s_mask, hsv_mask, mask=h_mask)
    return hsv_mask

def rg_filter(r, g):
    rg_sub = image_empty_clone(r)
    cv.Sub(r,g,rg_sub)
    rg_sub_thres = image_empty_clone(r)
    cv.Threshold(rg_sub,rg_sub_thres,1,255,cv.CV_THRESH_BINARY)

    rg_diff = image_empty_clone(r)
    cv.AbsDiff(r,g,rg_diff)
    rg_diff_thres = image_empty_clone(r)
    cv.Threshold(rg_diff,rg_diff_thres,11,255,cv.CV_THRESH_BINARY)

    res = image_empty_clone(r)
    cv.And(rg_diff_thres, rg_sub_thres, res)
    return res

def norm_rg_filter(r,g,b):
    nr, ng, _ = get_normalized_rgb_planes(r,g,b)

    nr_mask = get_filtered_plane(nr, ((0, 0.33), (0.6, 1)), 255, probability_to_255)
    ng_mask = get_filtered_plane(ng, ((0, 0.25), (0.37, 1)), 255, probability_to_255)

    res = image_empty_clone(nr)
    cv.And(nr_mask, ng_mask, res)
    return res

def skin_mask(img):
    r,g,b = get_rgb_planes(img)
    hsv_mask = filter_by_hsv(img, {
        "h": ((50, 340),),
        "s": ((0, 0.12), (0.7, 1),),
        "v": ((0, 0.3),)})

    rg_mask = rg_filter(r,g)
    nr_ng_mask = norm_rg_filter(r,g,b)

    tmp = image_empty_clone(hsv_mask)
    total_mask = image_empty_clone(hsv_mask)
    cv.And(hsv_mask,rg_mask,tmp)
    cv.And(tmp,nr_ng_mask,total_mask)

    th = image_empty_clone(total_mask)
    cv.Smooth(total_mask,total_mask,cv.CV_GAUSSIAN,11,11)
    cv.Threshold(total_mask, th, 25, 255, cv.CV_THRESH_BINARY)

    return th, hsv_mask, rg_mask, nr_ng_mask

def filter_skin(img):
    mask, _, _, _ = skin_mask(img)
    res = image_empty_clone(img)
    cv.And(img, merge_rgb(mask,mask,mask), res)
    return res

def main():
    cap = cv.CaptureFromCAM(0)
    while 1:
        img = scale_image(cv.QueryFrame(cap), 4)
#        img = cv.QueryFrame(cap)
        img = normalize(img,aggressive=0.001)
#        mask, hsv_mask, rg_mask, nr_ng_mask = skin_mask(img)
        skin = filter_skin(img)
        cv.ShowImage("cam", img)
        cv.ShowImage("skin", skin)
#        cv.ShowImage("hsv_mask", hsv_mask)
#        cv.ShowImage("rg_mask", rg_mask)
#        cv.ShowImage("nr_ng_mask", nr_ng_mask)
#        cv.ShowImage("mask", mask)
        key = cv.WaitKey(10)
        if key == 27:
            break

if __name__ == "__main__":
    main()