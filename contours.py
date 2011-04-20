import cv
import os

from cvutils import *
from skindetect import *
from utils import *
from mstclustering import merge_boxes

@time_took
def get_mask_with_contour(img, ret_img=False, ret_cont=False, with_init_mask=False,
                          cont_color=cv.RGB(255, 50, 50), normalize=True, skin_version=1, strong=False):
    if normalize:
        img = normalize_rgb(img, aggressive=0.005)
    mask = skin_mask(img) if skin_version == 1 else skin_mask2(img)

    di_mask = image_empty_clone(mask)
    cv.Dilate(mask, di_mask)

    seqs = cv.FindContours(cv.CloneImage(di_mask), memory(), cv.CV_RETR_EXTERNAL)

    c_img = image_empty_clone(mask)
    cv.DrawContours(c_img, seqs, 255, 255, 10, -1)

    er_img = image_empty_clone(c_img)
    cv.Erode(c_img, er_img,iterations=2)

    seqs = cv.FindContours(cv.CloneImage(er_img), memory(), cv.CV_RETR_EXTERNAL)
    if not seqs:
        print "no areas"
        return img, None
    seqs = cv.ApproxPoly(seqs, memory(),cv.CV_POLY_APPROX_DP,parameter=3,parameter2=1)

    result = []
    if ret_img:
#        er_seq_img = cv.CreateImage(sizeOf(er_img), 8, 3)
#        cv.Zero(er_seq_img)
        er_seq_img = cv.CloneImage(img)
        if with_init_mask:
            cv.Merge(mask,mask,mask,None,er_seq_img)

        if strong:
            cv.DrawContours(er_seq_img, seqs, cont_color, 0, 10, thickness=3)
            cv.DrawContours(er_seq_img, seqs, cv.RGB(0,0,0), 0, 10, thickness=1)
        else:
            cv.DrawContours(er_seq_img, seqs, cont_color, 0, 10, thickness=1)
        result.append(er_seq_img)

    if ret_cont:
        result.append(seqs)

    return result


def draw_convex_hull(seqs, img):
    seqs2 = cv.ConvexHull2(seqs, memory(), orientation=1, return_points=1)
    while seqs2:
        cv.DrawContours(img, seqs2, cv.RGB(50,255,50), cv.RGB(50,255,50), 10, 2)
        seqs2 = seqs2.v_next()

def seq_generator(seqs):
    while seqs:
        yield seqs
        seqs = seqs.h_next()

def is_inside(boxes, check_box):
    cx1,cy1,cw,ch = check_box
    cx2,cy2 = cx1+cw, cy1+ch
    for x1,y1,w,h in boxes:
        x2, y2 = x1+w, y1+h
        if cx1 == x1 and cx2 == x2 and cy1 == y1 and cy2 == y2:
            continue
        if cx1 >= x1 and cx2 <= x2 and cy1 >= y1 and cy2 <= y2:
            return True
    return False

def seqs_boxes(seqs, minsize=25):
    boxes = []
    new_seqs = []
    for seq in seqs:
        box = cv.BoundingRect(seq)
        if box[2] >= minsize and box[3] >= minsize and not is_inside(boxes, box):
            boxes.append(box)
            new_seqs.append(seq)
    boxes = [box for box in boxes if not is_inside(boxes, box)]
    return boxes, new_seqs

def seq_min_rects(seqs):
    min_rects = []
    for seq in seqs:
        min_rect = cv.MinAreaRect2(seq)
        min_rects.append(min_rect)
    return min_rects

def get_skin_rectangles(seqs, minsize=25):
    seqs_list = [seq for seq in seq_generator(seqs)]
    boxes, seqs_list = seqs_boxes(seqs_list, minsize)
    min_rects = seq_min_rects(seqs_list)
    return boxes, min_rects

@time_took
def _rect_stuff(seqs):
    boxes, min_rects = get_skin_rectangles(seqs)
    draw_boxes(boxes, img)
    polys = map(cv.BoxPoints, min_rects)
    cv.PolyLine(img, polys, True, cv.RGB(50,50,255), 2)
#    return boxes

@time_took
def _boxes(seqs,img,with_merge):
    boxes, min_rects = get_skin_rectangles(seqs)
    draw_boxes(boxes,img,color=cv.RGB(255,255,255),thickness=1)
    if with_merge:
        boxes = merge_boxes(boxes)
    draw_boxes(boxes,img)

def draw_face_contour_boxes(img, with_merge=True):
    img = normalize_rgb(img, aggressive=0.005)
    mask, seqs, time = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False, time_took=True)
    if not seqs:
        return img
#    _, time2 = _rect_stuff(seqs, mask, time_took=True)
    _,time2 = _boxes(seqs,mask,with_merge, time_took=True)
    write_info(mask, "%.6f face, %.6f boxes" % (time, time2))
    return mask

def _webcam_test():
    cap = cv.CaptureFromCAM(0)
    while 1:
        img = cv.QueryFrame(cap)
#        img = draw_face_contour_boxes(img)
        img = draw_face_contour_boxes(img)
        cv.ShowImage("mask", img)
#        cv.ShowImage("img", img)
        key = cv.WaitKey(10)
        if key == 27:
            break



def merge_images(img1, img2, vertical=None):
    w,h = sizeOf(img2)
    new_size, second_roi = ((w*2, h), (w,0,w,h)) if h*1.3 > w and not vertical else ((w, h*2), (0,h,w,h))
    merged = cv.CreateImage(new_size, img1.depth, img1.channels)
    cv.SetImageROI(merged, (0,0,w,h))
    cv.Copy(img1, merged)
    cv.SetImageROI(merged, second_roi)
    cv.Copy(img2, merged)
    cv.ResetImageROI(merged)
    return merged

def get_face_in_boxes(img):
    img = normalize_rgb(img, aggressive=0.005)
    mask, seqs, time = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False, time_took=True)
    boxes, min_rects = get_skin_rectangles(seqs,minsize=15)
    boxes = merge_boxes(boxes)
    return boxes

def _process_one_file(full_path, output_dir, name):
    img = scale_image(cv.LoadImage(full_path))
#            img = cv.LoadImage(full_path)
    img_with_cont = draw_face_contour_boxes(img)
#    merged = merge_images(img_with_cont, img)
    cv.SaveImage(os.path.join(output_dir, name), img_with_cont)

def batch_test():
#    path = "/Users/soswow/Documents/Face Detection/Frontal face dataset"
    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.02.25"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2010.12.22-1"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2010.12.22-2"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.01.26-1"
#    path = "/Users/soswow/Documents/Face Detection/Face Detection Data Set and Benchmark/originalPics/2002/07/19/big/"
#    path = "/Users/soswow/Documents/Face Detection/Face Detection Data Set and Benchmark/originalPics/2002/07/21/big/"
    output_dir = os.path.join(path, "cv")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for full_path, name in directory_files(path):
        print name
        try:
            _process_one_file(full_path, output_dir, name)
        except IOError:
            print "ignoring %s" % name
    print 'Done'

ft = "latex/Pictures/"
def main():
    img = cv.LoadImage(ft+"lesleythe-science-girl-7.jpg")
#    img = normalize_rgb(img, aggressive=0.002)
#    mask = skin_mask2(img)
    mask = get_mask_with_contour(img,ret_img=True)

#    img, skin = _main(img,version=2)
#    cv.SaveImage(ft+"dr_house_skin_mask_3.png", mask)
    show_images({"img":img, "mask":mask[0]})

if __name__ == "__main__":
#    main()
#    _webcam_test()
#    _process_one_file("/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.01.26-1/IMG_7348.JPG",
#    "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.01.26-1/cv","IMG_7348.JPG")
    batch_test()
#    img = cv.LoadImage("sample/img_563.jpg")
#    get_mask_with_contour(img)