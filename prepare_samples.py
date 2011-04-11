import math
import os
import cv
import shutil
import random
from os.path import join as pjoin

from sliding_window import *
from utils import *
from cvutils import *
from canny import get_canny_img

root_folder = "/Users/soswow/Documents/Face Detection/test/"

def get_prepared(img):
    if img.channels > 1:
        return prepare_bw(img)
    else:
        return normalize_plane(img,aggressive=0.005)


def wild_face():
    path = "/Users/soswow/Documents/Face Detection/lfwcrop_color/faces"
    positive_path=root_folder+"positive/wild"
    print  "Reading faces from %s" % path
    k=0
    for fullpath, name in directory_files(path):
        new_name = os.path.join(positive_path, name.split(".")[0] + ".png")
        img = scale_image(cv.LoadImage(fullpath))

        cv.SaveImage(new_name, get_prepared(img))
#        if k > n:
#            break
        k+=1
    print "%d samples added" % k

def crop_att_faces():
    path="/Users/soswow/Documents/Face Detection/att_faces/pgm"
    positive_path=root_folder+"positive/att_faces"
    k = 0
    for s in range(1,41):
        dir = os.path.join(path,"s%d" % s)
        for file, name in directory_files(dir):
            if file.endswith(".pgm"):
                k+=1
                img = cv.LoadImage(file)
                cv.SetImageROI(img, (7,32,78,78))
                dst = cv.CreateImage((32,32),8,3)
                cv.Resize(img, dst)
                cv.SaveImage(os.path.join(positive_path,"s%d-%s.png" % (s,name.split(".")[0])), get_prepared(dst))
    print "%d samples added" % k

def crop_georg_faces():
    path="/Users/soswow/Documents/Face Detection/Georgia Tech face database /cropped_faces/"
    positive_path=root_folder+"positive/georgia"
    k=0
    for file, name in directory_files(path):
        if file.endswith(".jpg"):
            k+=1
            img = cv.LoadImage(file)
            w,h = sizeOf(img)
            nw = nh = w-10
            y = h-nh-10
            x = 5
            cv.SetImageROI(img, (x,y,nw,nh))
            dst = cv.CreateImage((32,32),8,3)
            cv.Resize(img, dst)
            cv.SaveImage(os.path.join(positive_path,"%s.png" % (name.split(".")[0])), get_prepared(dst))
    print "%d samples added" % k


def prepare_negatives():
    path = root_folder+"negative_source/new2"
    negatives_path=root_folder+"negative"
    for fullpath, name in directory_files(path):
        try:
            big_img = cv.LoadImage(fullpath)
        except IOError:
            continue
        print "Getting samples from %s" % name
        no_suffix_name = name.split(".")[0]
        negatives_new_path = os.path.join(negatives_path, no_suffix_name)
        if not os.path.exists(negatives_new_path):
            os.mkdir(negatives_new_path)
        sample_gen = samples_generator(big_img,32,32,3,2,bw_from_v_plane=False)
        k=0
#        random_from_generator()
        for sample, _ in sample_gen:
            prepared = normalize_plane(sample)
            if cv.CountNonZero(prepared) > 0:
                new_name=os.path.join(negatives_new_path, "%d.png" % k)
                cv.SaveImage(new_name, prepared)
                k+=1
#            if k > 1500:
#                break
        print k

def make_sets_from_path(init_path, sets_path, n_in_each_set=2000, sets_n=4):
    dirs_n = len([dir for dir in os.listdir(init_path) if os.path.isdir(pjoin(init_path, dir))])

    print "Seeing %d dirs" % dirs_n

    for sett in range(1,sets_n+1):
        print "Making set #%d" % sett
        new_set_path = os.path.join(sets_path, "%d" % sett)
        if not os.path.exists(new_set_path):
            os.makedirs(new_set_path)

        per_folder_n = n_in_each_set / dirs_n

        copied = 0
        f = 0
        for folder in os.listdir(init_path):
            full_path = os.path.join(init_path, folder)
            if os.path.isdir(full_path):
                print "Processing folder %s ..." % folder,
                files = os.listdir(full_path)
                take_here = min(len(files), per_folder_n)
                random_selection = random.sample(files, take_here)
                for fullname in [os.path.join(full_path, filename) for filename in
                               random_selection]:
                    shutil.copy(fullname, os.path.join(new_set_path,
                                                       "%s-%s" % (folder, os.path.split(fullname)[1])))
                    copied += 1
                f+=1
                more = n_in_each_set - copied
                per_folder_n = more / (dirs_n-f) if f < dirs_n else more
            print " %d copied so far" % copied

def make_sets():
    negatives_path=root_folder+"negative"
    positive_path=root_folder+"positive"
    sets_path = root_folder+"sets"
    make_sets_from_path(positive_path,pjoin(sets_path,"positive"))
    make_sets_from_path(negatives_path,pjoin(sets_path,"negative"))

def clone_with_(func, folder="with_mask"):
    from_root = root_folder+""
    path_from=(root_folder+"positive",
                root_folder+"negative")
    clone_to=root_folder+folder

    for path in path_from:
        for root, dirs, files in os.walk(path):
            print "Copying form %s" % root
            right_part = root.split(from_root)[1]
            for file in files:
                orig_path = pjoin(root, file)
                new_dir = pjoin(clone_to, right_part)
                new_path = pjoin(new_dir, file)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                try:
                    img = cv.LoadImage(orig_path, iscolor=False)
                except IOError:
                    continue

                cv.SaveImage(new_path, func(img))

def clone_with_mask():
    mask = get_mask(32,32)
    def with_mask(img):
        masked = cv.CreateImage(sizeOf(img),8,1)
        cv.Zero(masked)
        cv.Copy(img, masked, mask=mask)
    clone_with_(with_mask, "with_mask")

def clone_with_edges():
    clone_with_(get_canny_img, "edge_view")

def main():
#    clone_with_mask()
#    make_sets()
#    clone_with_edges()
    prepare_negatives()
#    wild_face()
#    make_selection()
#    crop_att_faces()
#    crop_georg_faces()

if __name__=="__main__":
    main()