import cv
from cvutils import *
from utils import *
from haar_detect import find_faces

def main():
    tot, tru = 0, 0
    for k in range(1,21):
        path = "/Users/soswow/Documents/Face Detection/att_faces/pgm/s%d/" % k
        for path, filename in directory_files(path):
            try:
                img = cv.LoadImage(path, iscolor=False)
                img = normalize_plane(img, aggressive=0.05)
                faces = find_faces(img)
                if faces:
                    fx, fy, fw, fh = face = faces[0][0]
                    draw_boxes((face,), img, color=255, thickness=1, with_text=False)
                    cv.SetImageROI(img, face)
                    eye_blocks = ((fw/7, fh/4, fw/3, fh/4),
                                  (fw-(fw/7)-fw/3, fh/4, fw/3, fh/4)) #fw-10*fw/21-fw/7
                    draw_boxes(eye_blocks, img, color=0, thickness=1, with_text=False)
                    for x,y,w,h in eye_blocks:
                        cv.SetImageROI(img, (x+fx, y+fy, fw/3, fh/4))
                        show_image(img)
                    cv.ResetImageROI(img)
                    tru+=1
                    show_image(img)
                else:
                    pass
#                    show_image(img)
                tot+=1

#                show_image(img)
            except IOError:
                pass
    print "Total: %d, true: %d" % (tot, tru)

if __name__ == "__main__":
    main()
  