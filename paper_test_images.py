from __future__ import division

from contours import *
from cvutils import *
from mstclustering import draw_graph
from sliding_window import get_mask
from utils import *
import networkx as nx

def spanning_trees():
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.02.25"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2010.12.22-1"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2010.12.22-2"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.01.26-1"
    path = "/Users/soswow/Documents/Face Detection/Face Detection Data Set and Benchmark/originalPics/2002/07/22/big/"
    output_dir = os.path.join(path, "cv_spaning")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for full_path, name in directory_files(path):
        print name
        try:
            img = cv.LoadImage(full_path)
#            img = scale_image(img)
            img = normalize_rgb(img, aggressive=0.005)

            img, seqs = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False,cont_color=cv.RGB(20,20,20))
#            cv.DrawContours(img, seqs, cv. RGB(200,200,200), 0, 1, 1)
            boxes, _ = get_skin_rectangles(seqs)
#            img = image_empty_clone(img)
            draw_boxes(boxes, img, color=cv.RGB(255,255,255), thickness=1, with_text=False)
#            merged_boxes = merge_boxes(boxes)
#            draw_boxes(boxes,img)
            #    merged = merge_images(img_with_cont, img)

            merged_boxes = merge_boxes(boxes)

            draw_boxes(merged_boxes, img, color=cv.RGB(255,255,255), thickness=2, with_text=False)

            verticies = np.array([(x+w/2, y+h/2) for x,y,w,h in boxes])
            G = nx.complete_graph(len(verticies))
            for edge in G.edges():
                dist = np.linalg.norm(verticies[edge[0]]-verticies[edge[1]])
                nx.set_edge_attributes(G, "weight", {edge:dist})
            mst = nx.minimum_spanning_tree(G)

#            draw_graph(img,mst,verticies,color=cv.RGB(255,255,255), thickness=3)
#            for x,y in verticies:
#                cv.Circle(img, (x,y), 8, cv.RGB(255,255,255), thickness=-1)

            cv.SaveImage(os.path.join(output_dir, name), img)
        except IOError:
            print "ignoring %s" % name

def draw_mask():
    mask = get_mask(32,32)
##    kk1 = cv.CreateImage(sizeOf(mask), 8, 3)
#    w,h=32,32
#    img = cv.CreateImage((w,h), 8, 4)
#    cv.Zero(img)
#    t = int(w / 5)
#    k = int(w / 15)
#    p = w-k-1
#    poly =( (k,t), (t+k,0), (p-t,0), (p,t),
#            (p,h-t), (p-t, h), (t+k,h), (k,h-t))
#    cv.FillPoly(img,(poly,), 255)
##    cv.CvtColor(mask, kk1, cv.CV_GRAY2RGB)
##    cv.CvtColor(kk1, kk2, cv.CV_RGB2RGBA)
    cv.SaveImage("latex/Pictures/mask.png", mask)


def create_path(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def play_with_contrast():
    path = "latex/Pictures/contrast"

    path = "/Users/soswow/Documents/Face Detection/Face Detection Data Set and Benchmark/originalPics/2002/07"
#    path = "/Users/soswow/Documents/Face Detection/Face Detection Data Set and Benchmark/originalPics/2002/07/21/big/"
#    output_dir = "/Users/soswow/Documents/Face Detection/Face Detection Data Set and Benchmark/originalPics/2002/07/21/big_contrast/"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.02.25"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2010.12.22-1"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2010.12.22-2"
#    path = "/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.01.26-1"
    output_dir =  path+"_skin_draw/"
    create_path(output_dir)

    p=0
    font = cv.InitFont(cv.CV_FONT_HERSHEY_DUPLEX, 2, 2)
    for full_path, name in directory_files(path):
        print name
        base = name.split(".")[0]
        try:
            img = cv.LoadImage(full_path)
            if sizeOf(img)[0] > 1000:
                img = scale_image(img)
#            norm = normalize_rgb(img, aggressive=0)

#            cv.DrawContours(img, seqs, cv. RGB(200,200,200), 0, 1, 1)

#            skin1, seqs = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False,cont_color=cv.RGB(255,240,240))
#            skin2, seqs = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False,cont_color=cv.RGB(255,240,240),skin_version=2)
#
#            merged = merge_images(skin1, skin2)

#            merged = None
#            cv.SaveImage("%s%d_norm_no.png" % (output_dir,p), img)
#            for k in range(0, 11, 2):
#                aggres = k / 1000
#                norm = normalize_rgb(img, aggres)
#                norm, _ = get_mask_with_contour(norm, ret_cont=True, ret_img=True, with_init_mask=False, cont_color=cv.RGB(255,0,0), normilize=False)

#                write_info(norm, "%.3f" % aggres, font)
#                cv.SaveImage("%s%d_norm_%.3f.png" % (output_dir,p, aggres), norm)
            p+=1
#            cv.SaveImage("%s%s_005.png" % (output_dir,base), merge_images(img, norm_0052))
#            cv.SaveImage("%s%s_norm_row.png" % (output_dir,base), merged)
            img = draw_skin_boxes(img)
            cv.SaveImage("%s%d_skin.png" % (output_dir, p), img)
        except IOError:
            pass

def expand_box(box,pix):
    b = list(box)
    b[0]-=pix
    b[1]-=pix
    b[2]+=pix*2
    b[3]+=pix*2
    return tuple(b)


def draw_skin_boxes(img):
    img, seqs = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False,
                                      cont_color=cv.RGB(255, 240, 240), strong=False)
    boxes, _ = get_skin_rectangles(seqs)
    draw_boxes(boxes, img, thickness=1, color=cv.RGB(255, 0, 0), with_text=False)
    merged_boxes = merge_boxes(boxes, threshold=0.5)
    draw_boxes(map(lambda a: expand_box(a, 3), merged_boxes), img,
               thickness=1, color=cv.RGB(100, 100, 100), with_text=False)
    draw_boxes(map(lambda a: expand_box(a, 4), merged_boxes), img,
               thickness=1, color=cv.RGB(20, 255, 20), with_text=False)
    draw_boxes(map(lambda a: expand_box(a, 5), merged_boxes), img,
               thickness=1, color=cv.RGB(255, 255, 255), with_text=False)
    return img


def web_cam_recorder():
    cap = cv.CaptureFromCAM(0)
    path = "latex/Pictures/webcam/"
    create_path(path)
    files = sorted(os.listdir(path), reverse=True)
    try:
        k=int(files[0].split(".")[0])
    except Exception:
        k=0
    while True:
        img = cv.QueryFrame(cap)

        img = draw_skin_boxes(img)
        
        cv.ShowImage("window", img)
        key = cv.WaitKey(10)
        if key == 32:
            k+=1
            cv.SaveImage("%s%d.png" % (path, k), img)
        if key == 27:
            break

def main():
#    draw_mask()
    play_with_contrast()
#    web_cam_recorder()
#    spanning_trees()

if __name__ == "__main__":
    main()