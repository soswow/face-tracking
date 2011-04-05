import cv
from cvutils import *
from skindetect import *

@time_took
def get_mask_with_contour(img, ret_img=False, ret_cont=False, with_init_mask=False):
    img = normalize(img, aggressive=0.005)
    mask = skin_mask(img)

    di_mask = image_empty_clone(mask)
    cv.Dilate(mask, di_mask)

    seqs = cv.FindContours(cv.CloneImage(di_mask), memory(), cv.CV_RETR_EXTERNAL)

    c_img = image_empty_clone(mask)
    cv.DrawContours(c_img, seqs, 255, 255, 10, -1)

    er_img = image_empty_clone(c_img)
    cv.Erode(c_img, er_img,iterations=2)

    seqs = cv.FindContours(cv.CloneImage(er_img), memory(), cv.CV_RETR_EXTERNAL)
    seqs = cv.ApproxPoly(seqs, memory(),cv.CV_POLY_APPROX_DP,parameter=3,parameter2=1)

    result = []
    if ret_img:
        er_seq_img = cv.CreateImage(sizeOf(er_img), 8, 3)
        cv.Zero(er_seq_img)
        if with_init_mask:
            cv.Merge(mask,mask,mask,None,er_seq_img)

        cv.DrawContours(er_seq_img, seqs, cv.RGB(255,50,50), 0, 10, thickness=2)
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

def seqs_boxes(seqs):
    boxes = []
    new_seqs = []
    for seq in seqs:
        box = cv.BoundingRect(seq)
        if box[2] >= 25 and box[3] >= 25:
            boxes.append(box)
            new_seqs.append(seq)
    return boxes, new_seqs

def draw_boxes(boxes, img):
    for box in boxes:
        cv.Rectangle(img, box[:2], (box[2]+box[0], box[3]+box[1]), cv.RGB(50,255,50), thickness=2)

def seq_min_rects(seqs):
    min_rects = []
    for seq in seqs:
        min_rect = cv.MinAreaRect2(seq)
        min_rects.append(min_rect)
    return min_rects


@time_took
def _rect_stuff(seqs, img):
    seqs_list = [seq for seq in seq_generator(seqs)]
    boxes, seqs_list = seqs_boxes(seqs_list)
    draw_boxes(boxes, img)
    min_rects = seq_min_rects(seqs_list)
    polys = map(cv.BoxPoints, min_rects)
    cv.PolyLine(img, polys, True, cv.RGB(50,50,255), 2)

def main():
    cap = cv.CaptureFromCAM(0)
    while 1:
        img = cv.QueryFrame(cap)
        img = normalize(img, aggressive=0.005)
        mask, seqs, time = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False, time_took=True)
        _, time2 = _rect_stuff(seqs, mask, time_took=True)
        write_info(mask, "%.6f face, %.6f boxes" % (time, time2))

        cv.ShowImage("mask", mask)
        key = cv.WaitKey(10)
        if key == 27:
            break

if __name__ == "__main__":
    main()
#    img = cv.LoadImage("sample/img_563.jpg")
#    get_mask_with_contour(img)