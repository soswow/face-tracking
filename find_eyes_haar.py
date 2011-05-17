from math import sqrt
from haar_detect import find_eyes, find_mouth
from cvutils import *
from utils import *
from contours import seq_generator, seqs_boxes

def center(box):
    return box[0] + box[2]/2, box[1]+box[3]/2


class Face(object):
    def __init__(self, path):
        self.left_eye = None
        self.right_eye = None
        self.eyes_dark = []
        self.mouth = None

        img = cv.LoadImage(path, iscolor=False)
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
            #            show_image(thr_img)
            #            tmp = image_empty_clone(thr_img)
            #            cv.Erode(thr_img, tmp)
            #            show_image(tmp)

        seqs = cv.FindContours(cv.CloneImage(thr_img), memory(), cv.CV_RETR_EXTERNAL)
        boxes, new_seqs = seqs_boxes([seq for seq in seq_generator(seqs)], minsize=4)
        boxes.sort(reverse=True, key=lambda b:b[1]+b[3]/2)
#        img2 = image_empty_clone(thr_img)
#        cv.Zero(img2)
#        cv.DrawContours(img2, new_seqs[0], 255, 255, 10, -1)
        #            show_image(img2)
#        cv.Dilate(img2, img2)
        #            show_image(img2)
        self.eyes_dark.append(boxes[0])
        #        cv.Copy(thr_img, self.img)
        cv.ResetImageROI(self.img)


    def insert_eyes(self, eyes):
        left_eye, right_eye = [eye[0] for eye in eyes]
        if center(left_eye)[0] > center(right_eye)[0]:
            left_eye, right_eye = right_eye, left_eye
        self.left_eye, self.right_eye = left_eye, right_eye

    def eyes_centers(self):
        left = self._eye_center(self.left_eye)
        right = self._eye_center(self.right_eye)
        return left, right

    def choose_mouth(self, mouths):
        mouths.sort(reverse=True,key=lambda a:center(a[0])[1])
        self.mouth = mouths[0][0]

    def draw_face(self):
        img = merge_rgb(self.img, self.img, self.img)
        draw_boxes((self.left_eye, self.right_eye), img, color=(0,0,255), thickness=1, with_text=False)
        draw_boxes((self.mouth,), img, color=(255,0,0), thickness=2, with_text=False)
        for eye, dark in ((self.left_eye, self.eyes_dark[0]), (self.right_eye, self.eyes_dark[1])):
            cv.SetImageROI(img, eye)
            draw_boxes((dark, ),img,(0,255,0),thickness=1, with_text=False)
#            cv.Set(img, (0,255,0), dark)
            cv.ResetImageROI(img)
        return img

    def init_facial_features(self):
        eyes = find_eyes(self.img)
        mouths = find_mouth(self.img)
        if len(eyes) != 2 or not mouths:
            raise Exception("Not enough features")
        self.insert_eyes(eyes)
        self.choose_mouth(mouths)
        self.eyes_centers()

def find_in_db():
    for k in range(1,21):
        path = "/Users/soswow/Documents/Face Detection/att_faces/pgm/s%d/" % k
        for path, filename in directory_files(path):
            try:
                face = Face(path)
                show_image(face.draw_face())
            except Exception:
                pass

def main():
    find_in_db()
#    with_webcam(find_and_box_eyes)

if __name__ == "__main__":
    main()
  