from __future__ import division

import cv

from cvutils import *

def samples_generator(img, w, h, slide_step=1, resize_step=1.2):
    _,_,img = get_hsv_planes(img)
    img = normalize_plane(img)
    ow, oh = sizeOf(img)
    if ow < w and oh < h:
        raise Exception("Requested sample is bigger than source")

    cw, ch = ow*resize_step, oh*resize_step
    while cw > w and ch > h:
        ch /= resize_step
        cw /= resize_step

        tmp = cv.CreateImage((cw,ch), 8, 1)
        cv.Resize(img, tmp)
        img = tmp

        if cw < w or ch < h:
            cw,ch = w,h

        for cx in range(0,cw-w+1,slide_step):
            for cy in range(0,ch-h+1,slide_step):
                cv.SetImageROI(img, (cx,cy,w,h))
                yield img
        cv.ResetImageROI(img)

@time_took
def test_take_samples():
    img = cv.LoadImage("sample/lena.bmp")
    sample_gen = samples_generator(img, 50, 50, slide_step=4, resize_step=1.3)
    for i, sample in enumerate(sample_gen):
        pass

def profile():
#    import cProfile
#    import pstats
    pass
#    cProfile.run('take_samples()', 'take_samples')
#    p = pstats.Stats('take_samples')
#    p.strip_dirs().sort_stats('cumulative').print_stats()

def main():
    test_take_samples()

if __name__ == "__main__":
    main()