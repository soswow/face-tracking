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

def get_filtered_plane(plane, filter_ranges, ranges_func):
    mask = image_empty_clone(plane)
    for start, stop in filter_ranges:
        start, stop = ranges_func(start, stop) #0.3, 1.0
        tmp = image_empty_clone(plane)
        cv.InRangeS(plane, start, stop, tmp)
        cv.Or(mask, tmp, mask)
    return mask

def filter_by_hsv(img, ranges):
    h,s,v = get_hsv_planes(img)
    
    h_mask = get_filtered_plane(h, ranges["h"], h2cv_values)
    s_mask = get_filtered_plane(s, ranges["s"], probability_to_255)
    v_mask = get_filtered_plane(v, ranges["v"], probability_to_255)

    hsv_mask = image_empty_clone(v_mask)
    cv.And(v_mask, s_mask, hsv_mask, mask=h_mask)
    return hsv_mask

def first_bigger_then_second(a, b):
    sub = image_empty_clone(a)
    cv.Sub(a,b,sub) #Thouse with R < G will become 0
    binary = image_empty_clone(a)
    #Make binary image
    cv.Threshold(sub,binary,1,255,cv.CV_THRESH_BINARY)
    return binary

def abs_diff_threshold(a,b,level):
    abs_diff = image_empty_clone(a)
    cv.AbsDiff(a,b,abs_diff)
    binary = image_empty_clone(a)
    cv.Threshold(abs_diff,binary,level,255,cv.CV_THRESH_BINARY)
    return binary

def rg_filter(r, g, rg_diff=11, b=None):
    #Checking rule: R > G
    rg_sub_binary = first_bigger_then_second(r,g)

    #Checking rule: R > B
    rb_sub_binary= first_bigger_then_second(r,b) if b else None

    #Checking rule: |R - G| >= 11
    rg_diff_thres = abs_diff_threshold(r,g,rg_diff)

    res = image_empty_clone(r)
    cv.And(rg_diff_thres, rg_sub_binary, res)

    if rb_sub_binary:
        cv.And(res, rb_sub_binary, res)    
    return res

def norm_rg_filter(r,g,b):
    nr, ng, _ = get_normalized_rgb_planes(r,g,b)

    nr_mask = get_filtered_plane(nr, ((0.33, 0.6),), probability_to_255)
    ng_mask = get_filtered_plane(ng, ((0.25, 0.37),), probability_to_255)

    res = image_empty_clone(nr)
    cv.And(nr_mask, ng_mask, res)
    return res

@time_took
def skin_mask(img):
    r,g,b = get_rgb_planes(img)
    hsv_mask = filter_by_hsv(img, {
        "h": ((0,50), (340, 360)),
        "s": ((0.12, 0.7),),
        "v": ((0.3, 1),)})

    rg_mask = rg_filter(r,g)
    nr_ng_mask = norm_rg_filter(r,g,b)

    tmp = image_empty_clone(hsv_mask)
    total_mask = image_empty_clone(hsv_mask)
    cv.And(hsv_mask,rg_mask,tmp)
    cv.And(tmp,nr_ng_mask,total_mask)

    #TODO What is this?
    th = image_empty_clone(total_mask)
    cv.Smooth(total_mask,total_mask,cv.CV_MEDIAN, 5, 5)
    cv.Threshold(total_mask, th, 25, 255, cv.CV_THRESH_BINARY)

    return total_mask

def rgb_min_max_diff_plane(r,g,b,level):
    rg_max = image_empty_clone(r)
    cv.Max(r,g,rg_max)
    rgb_max = image_empty_clone(b)
    cv.Max(rg_max,b,rgb_max)

    rg_min = image_empty_clone(r)
    cv.Min(r,g,rg_min)
    rgb_min = image_empty_clone(b)
    cv.Min(rg_min,b,rgb_min)

    rgb_sub = image_empty_clone(rgb_max)
    cv.Sub(rgb_max, rgb_min, rgb_sub)

    binary = image_empty_clone(r)
    cv.Threshold(rgb_sub,binary,level,255,cv.CV_THRESH_BINARY)

    return binary

def and_planes(planes):
    assert len(planes) > 0
    res = planes[0]
    for plane in planes[1:]:
        cv.And(plane, res, res)
    return res

@time_took
def skin_mask2(img):
    r,g,b = get_rgb_planes(img)

    r_plane = get_filtered_plane(r, ((95, 255),), lambda s,f:(s,f))
    g_plane = get_filtered_plane(g, ((40, 255),), lambda s,f:(s,f))
    b_plane = get_filtered_plane(b, ((20, 255),), lambda s,f:(s,f))

    minmax_diff = rgb_min_max_diff_plane(r,g,b,16)
    rg_binary = rg_filter(r, g, 16, b=b)

    res = and_planes((r_plane,g_plane,b_plane,minmax_diff,rg_binary))
#    cv.Dilate(res, res)
#    th = image_empty_clone(res)
#    cv.Smooth(res,res,cv.CV_MEDIAN, 5, 5)

    return res

@time_took
def filter_skin(img, version=1):
    if version == 1:
        mask = skin_mask(img)
    elif version == 2:
        mask = skin_mask2(img)
    res = image_empty_clone(img)
    cv.Copy(img,res,mask)
#    cv.And(img, merge_rgb(mask,mask,mask), res)
    return res

@time_took
def _main(img, version=1):
#    img = scale_image(img, 4)
    img = normalize_rgb(img,aggressive=0.005)
    skin = filter_skin(img,version)
    return img, skin

def _webcam_test():
    cap = cv.CaptureFromCAM(0)
    while 1:
        img, cam_time = time_took(cv.QueryFrame)(cap, time_took=True)
        img_n = normalize_rgb(img, aggressive=0.003)
#        img = scale_image(img)
#        img, skin1, time = _main(img,version=1, time_took=True)
        skin1 = skin_mask(img_n)
#        _, skin2, _ = _main(img,version=2, time_took=True)
#        write_info(img, "%.6f cam, %.6f face" % (cam_time, time))
        cv.ShowImage("cam", img)
        cv.ShowImage("cam2", img_n)
        cv.ShowImage("skin1", skin1)
#        cv.ShowImage("skin2", skin2)

        key = cv.WaitKey(10)
        if key == 27:
            break

ft = "latex/Pictures/"
#ft = "sample/"
def main():
    img = cv.LoadImage(ft+"older-people-the-web1.jpg")
    img = normalize_rgb(img, aggressive=0.003)
    mask1 = skin_mask(img)
#    mask2 = skin_mask2(img)
#    mask = get_mask_with_contour(img)
#    img, skin = _main(img,version=2)
#    cv.SaveImage(ft+"dr_house_skin_mask_3.png", mask)
    show_images({"img":img, "mask":mask1})

if __name__ == "__main__":
    main()
#    _webcam_test()