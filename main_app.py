import cv
from contours import get_mask_with_contour, get_skin_rectangles
from cvutils import *
from laplace import *
from mstclustering import merge_boxes
from pybrain_utils import load_ann, get_flatten_image
from sliding_window import samples_generator

root_folder = "/Users/soswow/Documents/Face Detection/test/"

@time_took
def get_face_regions(ann, ann2, img, classifier):
    img = normalize_rgb(img, aggressive=0.005)
    mask, seqs = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False)
    if not seqs:
        return img
#    show_image(mask)
    skin_regions, min_rects = get_skin_rectangles(seqs)
    skin_regions = merge_boxes(skin_regions)
    draw_boxes(skin_regions,img,with_text=False,thickness=1)
#
#    small_img = prepare_bw(scale_image(img))
#    cv.EqualizeHist(small_img, small_img)
#    objects = cv.HaarDetectObjects(small_img, classifier, cv.CreateMemStorage(0), 1.1, 3,
#                         cv.CV_HAAR_DO_CANNY_PRUNING | cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv.CV_HAAR_DO_ROUGH_SEARCH,
#                         min_size=(50,50))
#    found = [[k*2 for k in obj] for obj, num in objects]
#    draw_boxes(found, img, with_text=False, color=cv.RGB(255,255,255))

    for region in skin_regions:
#        cv.SetImageROI(img, region)

#        cv.ResetImageROI(img)
#        if objects:



        cv.SetImageROI(img, region)
        region_img = cv.CreateImage(region[2:],img.depth,img.channels)
        cv.Copy(img, region_img)
        found = []
        try:
            for i,(sample, box) in enumerate(samples_generator(region_img, 32, 32, slide_step=4, resize_step=1.5, bw_from_v_plane=False)):
#                cv.SaveImage(root_folder+"webcam/%d.png" % (p+i), sample)
                nf, f = ann.activate(get_flatten_image(sample))
                nf2, f2 = ann2.activate(get_flatten_image(laplace(sample)))
                buf_nf, buf_f = tuple(ann['out'].inputbuffer[0])
                _, buf_f2 = tuple(ann2['out'].inputbuffer[0])
                if f > nf and f2 > nf2 and buf_f > 250000 and buf_f2 > 50000:
                    found.append(box)
        except Exception:
            pass
        if found:
            draw_boxes(found, img, with_text=False, color=cv.RGB(255,255,255))
        cv.ResetImageROI(img)
    return img

def webcam():
    cam = cv.CaptureFromCAM(0)
    ann = load_ann("default-ann7.9")
    ann2 = load_ann("default-ann")
    black = cv.RGB(0,0,0)
    p=0
    haar = cv.Load("haarcascade_frontalface_alt2.xml")
    while True:
#        if p > 3:
#            break
        img = scale_image(cv.QueryFrame(cam))
        img, time = get_face_regions(ann, ann2, img, haar, time_took=True)
        write_info(img, "%.6f canny" % time, color=black)
        cv.ShowImage("faces", img)
        key = cv.WaitKey(10)
        if key == 27:
            break


def main():
#    img = scale_image(cv.LoadImage("sample/Group-Oct06.jpg"))
#    img = get_face_regions(img)
#    show_image(img)
    webcam()

if __name__ == "__main__":
    main()