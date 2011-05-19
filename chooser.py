import cv
from genericpath import exists
from os import makedirs
from shutil import copy,move
from cvutils import sizeOf, show_image
from utils import yield_files_in_path

def hand_filtering(sub_base):
    base = "/Users/soswow/Documents/Face Detection/test/normalized/"
    i=0

    #220 - Andrew_Niccol_0002.jpg
    go = False
    for path, filename in yield_files_in_path(base+"raw/%s/big" % sub_base):
        try:
#            if not go and not filename.startswith("220 -"):
#                continue
#            else:
#                go = True
            img = cv.LoadImage(path, iscolor=False)
            cv.ShowImage("choose", img)
            key = cv.WaitKey(0)
            if key == 13:
                print "%d - %s" % (i, filename)
                i+=1
                copy(base+("raw/%s/small/" % sub_base) +filename, base+("chosen/%s/" % sub_base)+filename)
        except IOError, e:
            pass

def is_equalized(img):
    cv.SetImageROI(img, (2,0,3,20))
    left_avg = cv.Avg(img)[0]
    cv.ResetImageROI(img)

    cv.SetImageROI(img, (15,0,18,20))
    right_avg = cv.Avg(img)[0]
    cv.ResetImageROI(img)

    cv.SetImageROI(img, (5,0,10,20))
    mid_avg = cv.Avg(img)[0]
    cv.ResetImageROI(img)

    l_diff = abs(mid_avg - left_avg)
    r_diff = abs(mid_avg - right_avg)
    
#    print l_diff, r_diff

    if l_diff > 65 or r_diff > 65:
        return False
    elif abs(l_diff - r_diff) > 45:
        return False
    else:
        return True

def filter_by_intensity(sub_base):
    base = "/Users/soswow/Documents/Face Detection/test/normalized/chosen/%s/" % sub_base
    filter = {(0,111):"low/", (111, 181):"med/", (181, 255):"hi/"}
    for dir in filter.values() + ["bad/"]:
        create = base + dir
        if not exists(create):
            makedirs(create)

    for path, filename in yield_files_in_path(base):
        try:
            base = "/Users/soswow/Documents/Face Detection/test/normalized/chosen/%s/" % sub_base
            img = cv.LoadImage(path, iscolor=False)
            good = is_equalized(img)
            copied = False
            if not good:
                copied = True
                move(path, base+"bad/"+filename)
            else:
#                show_image(img)
                avg = cv.Avg(img)[0]
                for fr, to in filter.keys():
                    if avg >= fr and avg <= to:
                        copied = True
                        move(path, base+filter[(fr,to)]+filename)
                        break
            if not copied:
                print "hm", filename
#                print avg
        except IOError, e:
            print e, filename


def main():
#    hand_filtering("frontal")
    filter_by_intensity("frontal")

if __name__ == "__main__":
    main()