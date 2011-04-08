import os
import cv

from sliding_window import *
from utils import *
from cvutils import *

def prepare_positives():
    path = "/Users/soswow/Documents/Face Detection/lfwcrop_color/faces"
    positive_path="/Users/soswow/Documents/Face Detection/test/positive"
    print  "Reading faces from %s" % path
    k=0
    mask=get_mask(32,32)
    for fullpath, name in random_from_generator(directory_files(path), 35):
        new_name = os.path.join(positive_path, name.split(".")[0] + ".png")
        img = scale_image(prepare_bw(cv.LoadImage(fullpath)))
        masked=cv.CreateImage((32,32),8,1)
        cv.Zero(masked)
        cv.Copy(img, masked, mask)
        cv.SaveImage(new_name, masked)
        k+=1
    print "%d samples added" % k

def prepare_negatives():
    path = "/Users/soswow/Documents/Face Detection/test/negative_source"
    negatives_path="/Users/soswow/Documents/Face Detection/test/negative"
    for fullpath, name in directory_files(path):
        print "Getting samples from %s" % name
        no_suffix_name = name.split(".")[0]
        big_img = cv.LoadImage(fullpath)
        sample_gen = samples_generator(big_img,32,32,32,2)
        k=0
        for sample in random_from_generator(sample_gen, 10):
            prepared = normalize_plane(sample)
            if cv.CountNonZero(prepared) > 0:
                new_name=os.path.join(negatives_path, "%s-%d.png" % (no_suffix_name, k))
                cv.SaveImage(new_name, prepared)
                k+=1
            if k > 1500:
                break
        print k
#        new_name = os.path.join(positive_path, name.split(".")[0] + ".png")
#        img = scale_image(prepare_bw(cv.LoadImage(fullpath)))
#        masked=cv.CreateImage((32,32),8,1)
#        cv.Zero(masked)
#        cv.Copy(img, masked, mask)
#        cv.SaveImage(img, masked)

def main():
    prepare_negatives()


if __name__=="__main__":
    main()