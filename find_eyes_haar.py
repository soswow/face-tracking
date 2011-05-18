from __future__ import division
from math import sqrt
from haar_detect import find_eyes, find_mouth
from cvutils import *
from utils import *
from contours import seq_generator, seqs_boxes, contour_centroids

def get_center(box):
    return box[0] + box[2]/2, box[1]+box[3]/2

def euclid_distance(p1, p2):
    return sqrt(sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))

class Face(object):
    def __init__(self, path):
        self.left_eye = None
        self.right_eye = None
        self.eyes_centers_points = []
        self.mouth = None

        img = cv.LoadImage(path, iscolor=False)
        if img.channels > 1:
            img = black_and_white(img)
        img = normalize_plane(img, aggressive=0.05)
        self.img = img
        self.init_facial_features()

    def _eye_center(self, eye):
        cv.SetImageROI(self.img, eye)
        img = normalize_plane(self.img, aggressive=0.05)
        for th in range(0, 200, 5):
            thr_img = image_empty_clone(img)
            cv.Threshold(img, thr_img, th, 255, cv.THRESH_BINARY_INV)
            w,h = sizeOf(img)
            sum1 = cv.Sum(thr_img)[0]
            pers = sqrt(sum1 / (w * h * 255)) * 100
            if pers > 16:
                break

        seqs = cv.FindContours(cv.CloneImage(thr_img), memory(), cv.CV_RETR_EXTERNAL)
        centers = contour_centroids(seqs)
        cv.ResetImageROI(self.img)
        centers = [(x+eye[0], y+eye[1]) for x,y in centers]
        return centers


    def insert_eyes(self, eyes):
        left_eye, right_eye = [eye[0] for eye in eyes]
        if get_center(left_eye)[0] > get_center(right_eye)[0]:
            left_eye, right_eye = right_eye, left_eye
        self.left_eye, self.right_eye = left_eye, right_eye

    def filter_eye_centers(self):
        eyes = [self.left_eye, self.right_eye]
        min_j = [0,0]
        for ci, centers in enumerate(self.eyes_centers_points):
            eye_box_center = get_center(eyes[ci])
            min = 10000
            for j, center in enumerate(centers):
               dist = euclid_distance(eye_box_center, center)
               if dist < min:
                   min = dist
                   min_j[ci] = j
        for i, j in enumerate(min_j):
            self.eyes_centers_points[i] = self.eyes_centers_points[i][j]

    def eyes_centers(self):
        left = self._eye_center(self.left_eye)
        right = self._eye_center(self.right_eye)
        self.eyes_centers_points = [left, right]
        self.filter_eye_centers()

    def choose_mouth(self, mouths):
        mouths.sort(reverse=True,key=lambda a:get_center(a[0])[1])
        self.mouth = mouths[0][0]

    def draw_face(self):
        img = merge_rgb(self.img, self.img, self.img)
        draw_boxes((self.left_eye, self.right_eye), img, color=(0,0,255), thickness=1, with_text=False)
        draw_boxes((self.mouth,), img, color=(255,0,0), thickness=2, with_text=False)
        for center in self.eyes_centers_points:
            cv.Circle(img, center, 3, (125,255,0),thickness=-1)
        triangle = self.eyes_centers_points + [get_center(self.mouth)]
        cv.PolyLine(img, [triangle], True, (0,255,255))
        return img

    def init_facial_features(self):
        eyes = find_eyes(self.img)
        mouths = find_mouth(self.img)
        if len(eyes) != 2 or not mouths:
            raise Exception("Not enough features")
        self.insert_eyes(eyes)
        self.choose_mouth(mouths)
        self.eyes_centers()

    def rotate_face(self):
        l, r = map(list, self.eyes_centers_points)
        avg_eyes_y = (l[1] + r[1]) / 2
        l[1] = r[1] = avg_eyes_y
        mouth_center = get_center(self.mouth)
        mouth = (l[0] + ((r[0] - l[0])/2), mouth_center[1])
        from_triang = tuple(self.eyes_centers_points + [mouth_center])
        to_triang = tuple(map(tuple, [l, r, mouth]))

        img = merge_rgb(self.img, self.img, self.img)
        cv.PolyLine(img, [from_triang], True, (0,255,255))
        cv.PolyLine(img, [to_triang], True, (0,255,0))
        show_image(img)

        mapp = cv.CreateMat(2, 3, cv.CV_32F)
        cv.GetAffineTransform(from_triang, to_triang, mapp)
        dst = image_empty_clone(self.img)
        cv.WarpAffine(self.img, dst, mapp, cv.CV_WARP_FILL_OUTLIERS, 0)
        show_image(dst)

    def crop_face(self):
        pass

def paths_1():
    for k in range(1,21):
        yield "/Users/soswow/Documents/Face Detection/att_faces/pgm/s%d/" % k

def paths_2():
    yield "/Users/soswow/Documents/Face Detection/lfwcrop_color"

def find_in_db():
    for path in paths_1():
        for path, filename in directory_files(path):
            try:
                face = Face(path)
                show_image(face.draw_face())
                face.rotate_face()
            except Exception, e:
                pass

def main():
    find_in_db()
#    with_webcam(find_and_box_eyes)

if __name__ == "__main__":
    main()
  