import cv
import numpy as np

def show_image(img, window_name="win"):
    cv.ShowImage(window_name, img)
    cv.WaitKey(0)

def show_images(images_dict):
    for name, img in images_dict.items():
        cv.ShowImage(name, img)
    cv.WaitKey(0)

def image_empty_clone(img, size=None, channels=None):
    new_img = cv.CreateImage(size or cv.GetSize(img), img.depth, channels or img.channels)
    cv.Zero(new_img)
    return new_img

def get_three_planes(src):
    channels = [None] * 4
    size = cv.GetSize(src)
    for i in range(src.channels):
        channels[i] = cv.CreateImage(size, src.depth, 1)
    cv.Split(src,channels[0],channels[1],channels[2],None)
    return channels[:3]

def get_rgb_planes(img):
    b,g,r = get_three_planes(img)
    return r,g,b

def get_hsv_planes(img):
    new_img = image_empty_clone(img)
    cv.CvtColor(img, new_img, cv.CV_BGR2HSV)
    h,s,v = get_three_planes(new_img)
#    inv_s = image_empty_clone(img, channels=1)
#    cv.SubRS(s,255,inv_s)
    return h,s,v

def get_ycrcb_planes(img):
    new_img = image_empty_clone(img)
    cv.CvtColor(img, new_img, cv.CV_RGB2YCrCb)
    y,cr,cb = get_three_planes(new_img)
    return y,cr,cb

def merge_rgb(r,g,b):
    dst = cv.CreateImage(cv.GetSize(r),8,3)
    cv.Merge(b,g,r,None,dst)
    return dst


def scale_image(img_orig, scale_factor=2):
    orig_size = cv.GetSize(img_orig)
    new_size = (orig_size[0] / scale_factor, orig_size[1] / scale_factor)
    img = image_empty_clone(img_orig,size=new_size)
    cv.Resize(img_orig, img)
    return img

def normalize_plane(plane, aggressive=0):
    if aggressive:
#        smooth = image_empty_clone(plane)
#        cv.Smooth(plane, smooth, cv.CV_GAUSSIAN, 13, 13)
        hist = get_gray_histogram(plane, bins=255)
        _, max_value, _, max_color = cv.GetMinMaxHistValue(hist)
        thr_value = max_value * aggressive
        down_threshold, up_threshold = None, None
        for k in range(255):
            down_val = cv.QueryHistValue_1D(hist, k)
            up_val = cv.QueryHistValue_1D(hist, 254-k)
            if not down_threshold and down_val >= thr_value:
                down_threshold = k
            if not up_threshold and up_val >= thr_value:
                up_threshold = k
            if down_threshold and up_threshold:
                break
        sub_plane = image_empty_clone(plane)
        add_plane = image_empty_clone(plane)
        cv.SubS(plane, down_threshold, sub_plane)
        cv.AddS(sub_plane, down_threshold+up_threshold, add_plane)
#        cv.AddS(plane, down_color, sub_plane)
#        cv.Threshold(plane,sub_plane,down_color,down_color,cv.CV_THRESH_)
        plane = add_plane
    norm_plane = image_empty_clone(plane)
    cv.Normalize(plane, norm_plane, 0, 255, cv.CV_MINMAX)
    return norm_plane

def normalize(img, aggressive=0.005):
    rgb = get_rgb_planes(img)
    out_rgb = []
    for plane in rgb:
        out_rgb.append(normalize_plane(plane, aggressive))
    return merge_rgb(out_rgb[0],out_rgb[1],out_rgb[2])

def equalize(img):
    rgb = get_rgb_planes(img)
    out_rgb = []
    for plane in rgb:
        equal_plane = image_empty_clone(plane)
        cv.EqualizeHist(plane, equal_plane)
        out_rgb.append(equal_plane)
    return merge_rgb(out_rgb[0],out_rgb[1],out_rgb[2])

def get_gray_histogram(layer, bins=40, range_max=255):
    hist = cv.CreateHist([bins], cv.CV_HIST_ARRAY, [(0,range_max)], 1)
    cv.CalcHist([layer], hist)
    return hist

def get_hist_image(hist, bins, width=500):
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
    x = 0
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

def get_2d_hist(planes, x_bins, y_bins, x_range_max=255, y_range_max=255):
    x_ranges = [0, x_range_max]
    y_ranges = [0, y_range_max]
    ranges = [x_ranges, y_ranges]
    hist = cv.CreateHist([x_bins, y_bins], cv.CV_HIST_ARRAY, ranges, 1)
    cv.CalcHist(planes, hist)
    return hist

def get_hs_2d_hist(img, h_bins, s_bins):
    h_plane, s_plane, _ = get_hsv_planes(img)
    return get_2d_hist([h_plane, s_plane], h_bins, s_bins, x_range_max=180)

def get_2d_hist_img(x_bins=30, y_bins=32, scale=10, hist=None, img=None):
    if not hist:
        if img:
            hist = get_hs_2d_hist(img, x_bins, y_bins)
        else:
            raise Exception("Histogram or image should be given")

    (_, max_value, _, _) = cv.GetMinMaxHistValue(hist)

    hist_img = cv.CreateImage((x_bins*scale, y_bins*scale), 8, 3)

    for h in range(x_bins):
        for s in range(y_bins):
            bin_val = cv.QueryHistValue_2D(hist, h, s)
            intensity = cv.Round(bin_val * 255 / max_value)
            cv.Rectangle(hist_img,
                         (h*scale, s*scale),
                         ((h+1)*scale - 1, (s+1)*scale - 1),
                         cv.RGB(255-intensity, 255-intensity, 255-intensity),
                         cv.CV_FILLED)
    return hist_img

def cv2array(im):
  depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

  a = np.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
  a.shape = (im.height,im.width,im.nChannels)
  return a

def array2cv(a):
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im

def normalized_rgb(r,g,b):
    rgb_sum=r+g+b
    sr = r /rgb_sum
    sg = g /rgb_sum
    sb = b /rgb_sum
    return sr, sg, sb

def get_normalized_rgb_planes(r,g,b):
    size = cv.GetSize(r)
#    r,g,b = get_three_planes(img)

    nr_plane = cv.CreateImage(size,8, 1)
    ng_plane = cv.CreateImage(size,8, 1)
    nb_plane = cv.CreateImage(size,8, 1)

    r32 = cv.CreateImage(size,cv.IPL_DEPTH_32F, 1)
    g32 = cv.CreateImage(size,cv.IPL_DEPTH_32F, 1)
    b32 = cv.CreateImage(size,cv.IPL_DEPTH_32F, 1)
    sum = cv.CreateImage(size,cv.IPL_DEPTH_32F, 1)
    cv.Zero(sum)
    cv.Convert(r,r32)
    cv.Convert(g,g32)
    cv.Convert(b,b32)

    cv.Add(r32, g32, sum)
    cv.Add(b32, sum, sum)

    tmp = cv.CreateImage(size,cv.IPL_DEPTH_32F, 1)
    cv.Div(r32, sum, tmp)
    cv.ConvertScale(tmp, nr_plane, scale=255)
    cv.Div(g32, sum, tmp)
    cv.ConvertScale(tmp, ng_plane, scale=255)
    cv.Div(b32, sum, tmp)
    cv.ConvertScale(tmp, nb_plane, scale=255)

#    res = image_empty_clone(img)
#    cv.Merge(nr_plane,ng_plane,nb_plane,None,res)
    return nr_plane, ng_plane, nb_plane
#    return res
