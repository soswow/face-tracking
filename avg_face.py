import cv
import os.path as p
import os

from cvutils import *
from utils import *

def main():
#    root = "/Users/soswow/Documents/Face Detection/test/negative"
    sub_base = "frontal"
    root = "/Users/soswow/Documents/Face Detection/test/normalized/chosen/" + sub_base
#    root = "/Users/soswow/Documents/Face Detection/test/sets/negative"
#    root = "/Users/soswow/Documents/Face Detection/test/edge_view/positive"
#    root = "/Users/soswow/Documents/Face Detection/test/sobel/positive"
#    root = "/Users/soswow/Documents/Face Detection/test/sets/positive"
#    root = "/Users/soswow/Documents/Face Detection/test/falses"
    w,h = (20,20) #(32,32)
    for folder in os.listdir(root):
        path = p.join(root, folder)
        if p.isdir(path):
            sum = cv.CreateMat(w,h, cv.CV_32F)
            cv.Zero(sum)
            k = 0
            for path, _ in directory_files(path):
                try:
                    img = cv.LoadImage(path,iscolor=False)
                except IOError:
                    continue
                mat = cv.CreateMat(w,h,cv.CV_32F)
                cv.Zero(mat)
                cv.Convert(cv.GetMat(img), mat)
                cv.Add(mat,sum,sum)
                k+=1
            avg = cv.CreateMat(w,h, cv.CV_32F)
            cv.Zero(avg)
            count = cv.CreateMat(w,h,cv.CV_32F)
            cv.Zero(count)
            cv.Set(count, k)
            cv.Div(sum,count,avg)
            new_img = cv.CreateImage((w,h), 8, 0)
            cv.Zero(new_img)
            cv.Convert(avg, new_img)
            cv.SaveImage(p.join(root,"%s-%s-avg.png" % (folder,sub_base)), new_img)

if __name__ == "__main__":
    main()