import cv
from cvutils import *
from contours import *
import numpy as np

def main():
    img = cv.LoadImage("latex/Pictures/IMG_7324.CR2.jpg")
    img = normalize_rgb(img, aggressive=0.005)
    mask, seqs, time = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False, time_took=True)
    boxes, min_rects = get_skin_rectangles(seqs)
    draw_boxes(boxes, img)
    center = [(c[0],c[1]) for c,_,_ in min_rects]
    verticies = []
    for x,y,w,h in boxes:
        verticies+=[(x,y), (x+w,y),(x,y+h),(x+w,y+h)]

#    verticies = [(x+w/2, y+h/2) for x,y,w,h in boxes]

    polys = map(cv.BoxPoints, min_rects)
    cv.PolyLine(img, polys, True, cv.RGB(50,50,255), 2)

    sample_count = len(verticies)
    samples = cv.CreateMat(sample_count, 1, cv.CV_32FC2)
    clusters = cv.CreateMat(sample_count, 1, cv.CV_32SC1)
    [cv.Set1D(samples, i, verticies[i]) for i in range(sample_count)]
    cv.KMeans2(samples, 3, clusters,
                   (cv.CV_TERMCRIT_EPS + cv.CV_TERMCRIT_ITER, 10, 1.0))
    color = [cv.RGB(255,10,10),
             cv.RGB(255,255,10),
             cv.RGB(10,255,255),
             cv.RGB(255,10,255)]
    for i, xy in enumerate(verticies):
        cv.Circle(img, xy, 5, color[int(clusters[i,0])], thickness=-1)

#    np_centers = np.asarray(verticies)
#    result = cv.kmeans(verticies, 2, 0, 5, cv.CV_TERMCRIT_ITER)

    show_image(img)



if __name__ == "__main__":
    main()