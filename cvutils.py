import cv

def show_image(img, window_name="win"):
    cv.ShowImage(window_name, img)
    cv.WaitKey(0)

def show_images(images_dict):
    for name, img in images_dict.items():
        cv.ShowImage(name, img)
    cv.WaitKey(0)

def image_empty_clone(img, size=None, channels=None):
    new_img = cv.CreateImage(size or cv.GetSize(img), img.depth, channels or img.channels)
    return new_img

def get_three_planes(src):
    channels = [None] * 4
    size = cv.GetSize(src)
    for i in range(src.channels):
        channels[i] = cv.CreateImage(size, src.depth, 1)
    cv.Split(src,channels[0],channels[1],channels[2],None)
    return channels[:3]

def get_rgb_planes(img):
    r,g,b = get_three_planes(img)
    return r,g,b

def get_hsv_planes(img):
    new_img = image_empty_clone(img)
    cv.CvtColor(img, new_img, cv.CV_RGB2HSV)
    h,s,v = get_three_planes(new_img)
#    inv_s = image_empty_clone(img, channels=1)
#    cv.SubRS(s,255,inv_s)
    return h,s,v

def get_ycrcb_planes(img):
    new_img = image_empty_clone(img)
    cv.CvtColor(img, new_img, cv.CV_RGB2YCrCb)
    y,cr,cb = get_three_planes(new_img)
    return y,cr,cb

def scale_image(img_orig, scale_factor=2):
    orig_size = cv.GetSize(img_orig)
    new_size = (orig_size[0] / scale_factor, orig_size[1] / scale_factor)
    img = image_empty_clone(img_orig,size=new_size)
    cv.Resize(img_orig, img)
    return img

def normalize(img):
    dst = image_empty_clone(img)
    rgb = get_rgb_planes(img)
    out_rgb = []
    for plane in rgb:
        norm_plane = image_empty_clone(plane)
        cv.Normalize(plane, norm_plane, 0, 255, cv.CV_MINMAX)
        out_rgb.append(norm_plane)
    cv.Merge(out_rgb[0],out_rgb[1],out_rgb[2],None,dst)
    return dst

def get_gray_histogram(layer, bins=40):
    hist = cv.CreateHist([bins], cv.CV_HIST_ARRAY, [(0,255)], 1)
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