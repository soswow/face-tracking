from __future__ import division
from math import sqrt, atan, degrees
from haar_detect import find_faces, find_eyes, find_mouth
from cvutils import *
from utils import *
from contours import seq_generator, seqs_boxes, contour_centroids

def get_center(box):
    return box[0] + box[2]/2, box[1]+box[3]/2

def euclid_distance(p1, p2):
    return sqrt(sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))



class Face(object):
    def __init__(self, path, with_face_detection=True):
        self.left_eye = None
        self.right_eye = None
        self.eyes_centers_points = []
        self.eyes = []
        self.mouth = None
        self.imgs = []
        self.orig_imgs = []

        self.eye_threshold = 25

        img = cv.LoadImage(path, iscolor=False)
        if img.channels > 1:
            img = black_and_white(img)
        img = normalize_plane(img, aggressive=0.05)
        self.imgs.append(img)
        self.orig_img = cv.CloneImage(img)
        if with_face_detection:
            self.find_faces(img)
        else:
            self.orig_imgs.append(cv.CloneImage(img))
        self.init_facial_features()

    def find_faces(self, img):
        faces = find_faces(img)
        self.imgs = []
        self.orig_imgs = []
        for face, _ in faces:
#            face = faces[0][0]
            tmp = cv.CreateImage(face[2:4],8,1)
            cv.SetImageROI(img, face)
            cv.Copy(img, tmp)
            cv.ResetImageROI(img)

            self.imgs.append(tmp)
            self.orig_imgs.append(cv.CloneImage(tmp))

    def _eye_center(self, eye):
        cv.SetImageROI(self.img, eye)
        img = normalize_plane(self.img, aggressive=0.05)
        cv.Copy(img, self.img)
        for th in range(0, 200, 5):
            thr_img = image_empty_clone(img)
            cv.Threshold(img, thr_img, th, 255, cv.THRESH_BINARY_INV)
            w,h = sizeOf(img)
            sum1 = cv.Sum(thr_img)[0]
            pers = sqrt(sum1 / (w * h * 255)) * 100
            if pers > self.eye_threshold:
                break

        seqs = cv.FindContours(cv.CloneImage(thr_img), memory(), cv.CV_RETR_EXTERNAL)
        centers = contour_centroids(seqs)
        cv.ResetImageROI(self.img)
        centers = [(x+eye[0], y+eye[1]) for x,y in centers]
        return centers, seqs


    def insert_eyes(self, eyes):
        left_eye, right_eye = [eye[0] for eye in eyes]
        if get_center(left_eye)[0] > get_center(right_eye)[0]:
            left_eye, right_eye = right_eye, left_eye
        self.left_eye, self.right_eye = left_eye, right_eye
        self.eyes = [self.left_eye, self.right_eye]

    def filter_eye_centers(self):
        min_j = [0,0]
        for ci, centers in enumerate(self.eyes_centers_points):
            eye_box_center = get_center(self.eyes[ci])
            min = 10000
            for j, center in enumerate(centers):
                dist = euclid_distance(eye_box_center, center)
                if dist < min:
                    min = dist
                    min_j[ci] = j
        for i, j in enumerate(min_j):
            self.eyes_centers_points[i] = self.eyes_centers_points[i][j]

    def eyes_centers(self):
        left_c, left_seqs = self._eye_center(self.left_eye)
        right_c, right_seqs = self._eye_center(self.right_eye)
        self.eyes_centers_points = [left_c, right_c]
        self.eyes_contours = [left_seqs, right_seqs]
        self.filter_eye_centers()

    def choose_mouth(self, mouths):
        mouths.sort(reverse=True,key=lambda a:get_center(a[0])[1])
        self.mouth = mouths[0][0]

    def _draw_cross(self, img, p, size=5, color=(255,0,0),thickness=1):
        color = cv.RGB(*color)
        cv.Line(img, (p[0], p[1]-size), (p[0], p[1]+size), color,thickness=thickness)
        cv.Line(img, (p[0]-size, p[1]), (p[0]+size, p[1]), color,thickness=thickness)

    def draw_face(self):
        img = merge_rgb(self.img, self.img, self.img)
        for i, seq in enumerate(self.eyes_contours):
            eye = self.eyes[i]
            cv.SetImageROI(img, eye)
            cv.DrawContours(img, seq, (0,255,0), 0, 10, thickness=-1)
            cv.ResetImageROI(img)
        draw_boxes((self.left_eye, self.right_eye), img, color=(255,0,0), thickness=1, with_text=False)
        draw_boxes((self.mouth,), img, color=(0,0,255), thickness=2, with_text=False)

        triangle = self.eyes_centers_points + [get_center(self.mouth)]
        cv.PolyLine(img, [triangle], True, cv.RGB(255,255,0))
        for center in self.eyes_centers_points:
            self._draw_cross(img, center, size=4, thickness=1)
        
        return img

    def init_facial_features(self):
        self.img = None
        for img, origin in zip(self.imgs, self.orig_imgs):
            eyes = find_eyes(img)
            mouths = find_mouth(img)
            if len(eyes) != 2 or not mouths:
                continue
            else:
                self.img = img
                self.orig_img = origin
                break
        if not self.img:
            raise Exception("Not enough features")
        self.insert_eyes(eyes)
        self.choose_mouth(mouths)
        self.eyes_centers()

    def normalize_face(self, rotate=True, screw_mouth=False):
        (lx,ly), (rx, ry) = map(list, self.eyes_centers_points)
        dy = ry - ly
        dx = rx - lx
        dst = self.orig_img
        if rotate:
            eyes_degrees = degrees(atan(dy / dx))
            rot_mapp = cv.CreateMat(2, 3, cv.CV_32F)
            cv.GetRotationMatrix2D((lx,ly), eyes_degrees, 1, rot_mapp)

            size = sizeOf(self.img)
            dst = image_empty_clone(self.img, size=(size[0]*1.1, size[1]*1.1))
            cv.WarpAffine(self.orig_img, dst, rot_mapp, cv.CV_WARP_FILL_OUTLIERS, 0)
#            show_image(dst)
            self.img = dst

        if screw_mouth:
            r = (lx + dx,ly)
            l = (lx, ly)
            mouth_center = get_center(self.mouth)
            mouth = (l[0] + ((r[0] - l[0])/2), mouth_center[1])
            from_triang = (l, r, mouth_center)
            to_triang = tuple(map(tuple, [l, r, mouth]))

            tmp = merge_rgb(dst, dst, dst)
            cv.PolyLine(tmp, [from_triang], True, (0,255,255))
            cv.PolyLine(tmp, [to_triang], True, (0,255,0))
#            show_image(tmp)

            screw_mapp = cv.CreateMat(2, 3, cv.CV_32F)
            cv.GetAffineTransform(from_triang, to_triang, screw_mapp)
            tmp = image_empty_clone(dst)
            cv.WarpAffine(dst, tmp, screw_mapp, cv.CV_WARP_FILL_OUTLIERS, 0)
#            show_image(tmp)
            self.img = tmp

    def crop_face(self, draw_grid=True):
        p_top, p_bot, p_left, p_right = 26,0,23,23
        t = min([eye_c[1] for eye_c in self.eyes_centers_points]) - p_top
        b = self.mouth[1] + self.mouth[3] + p_bot
        l = self.eyes_centers_points[0][0] - p_left
        r = self.eyes_centers_points[1][0] + p_right

        w,h = sizeOf(self.img)
        l = 0 if l < 0 else l
        t = 0 if t < 0 else t
        rect = [l, t,
                w-l if r > w else r-l,
                h-t if b > h else b-t]

#        diff = rect[3]-rect[2]
#        if diff > 0:
#            rect[1] += abs(diff) / 2
#            rect[3] -= abs(diff)
#            print "Stripped vertically - %d" % abs(diff)
#        if h > w:
#            rect[1] += diff/2
#            rect[3] -= diff
#        elif w > h:
#            rect[0] += diff/2
#            rect[2] -= diff
        tmp = image_empty_clone(self.img, size=rect[2:])
        cv.SetImageROI(self.img, tuple(rect))

        cv.Copy(self.img, tmp)
        cv.ResetImageROI(self.img)
        rect[3] = rect[2]
        resize = image_empty_clone(tmp, size=rect[2:])
        cv.Resize(tmp, resize, cv.CV_INTER_LINEAR)
        tmp = resize

#        print sizeOf(tmp)
        if draw_grid:
            cv.Line(tmp,(p_left,0), (p_left, rect[3]), color=255)
            cv.Line(tmp,(rect[2]-p_right,0), (rect[2]-p_right, rect[3]), color=255)
            cv.Line(tmp,(0,p_top), (rect[2], p_top), color=255)

        return tmp

def paths_1():
    for k in range(1,41):
        pathhh = "/Users/soswow/Documents/Face Detection/att_faces/pgm/s%d/" % k
        for res in directory_files(pathhh):
            yield res

def paths_2():
    for res in directory_files("/Users/soswow/Documents/Face Detection/lfwcrop_color"):
        yield res

def paths_3():
    for tmp in yield_files_in_path("/Users/soswow/Documents/Face Detection/lfw"):
        yield tmp

def paths_4():
    for tmp in yield_files_in_path("/Users/soswow/Documents/Face Detection/Georgia Tech face database/cropped_faces"):
        yield tmp

def paths_5():
    for tmp in yield_files_in_path("/Users/soswow/Documents/Face Detection/Frontal face dataset"):
        yield tmp

def find_in_db():
#    base_url = "/Users/soswow/Documents/Face Detection/test/normalized/raw/wild"
    base_url = "/Users/soswow/Documents/Face Detection/test/normalized/raw/frontal/"
    i=0
    for path, filename in paths_5():
        try:
            filename = filename.replace(".pgm",".png")
            filename = filename.replace(".jpg",".png")
            filename = "%d - %s" % (i, filename)
            face = Face(path, with_face_detection=True)
#            show_image(face.draw_face())
            face.normalize_face()
            img = face.crop_face(draw_grid=True)
            cv.SaveImage(base_url + "big/" + filename, img)
#            show_image(img)
            img = face.crop_face(draw_grid=False)
            img = normalize_plane(img)
            resize = image_empty_clone(img, size=(20,20))
            cv.Resize(img, resize, cv.CV_INTER_LINEAR)
            img = resize
            cv.SaveImage(base_url + "small/" + filename, img)
            print filename
            i+=1
#            show_image(img)
#            face.img = res
#            face.init_facial_features()
#            show_image(face.draw_face())
        except Exception, e:
            pass
#            print e

def main():
    find_in_db()
#    with_webcam(find_and_box_eyes)

if __name__ == "__main__":
    main()
  