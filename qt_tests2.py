import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PIL import Image
from StringIO import StringIO
import cv
import time

def get_pil_green_image(w, h):
    clr = chr(0)+chr(255)+chr(0)
    im = Image.fromstring("RGB", (w,h), clr*(w*h))
    return im

def get_pil_image(path="sample/img_563.jpg"):
    im = Image.open(path)
    return im

def pil2qimage(pil_image, qimage):
    file = StringIO()
    pil_image.save(file, "BMP")
    qimage.loadFromData(file.getvalue(), "BMP")
    return qimage

app = QApplication(sys.argv)

class WebCamDialog(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.lock = QReadWriteLock()

        self.setGeometry(300, 300, 250, 150)

        self.webCamThread = WebCamThread(self.lock)

        button = QPushButton(text="Push me")
        self.connect(button, SIGNAL("clicked()"), self.clicked)
        self.frame = QLabel()
        self.frame.setFrameStyle(QFrame.Box)
        self.img = QImage()
        self.connect(self.webCamThread, SIGNAL("update_pic()"), self.update_pic)

        layout = QVBoxLayout()
        layout.addWidget(self.frame)
        layout.addWidget(button)

        self.setLayout(layout)

        self.setWindowTitle('Window')

    def update_pic(self):
        self.frame.setPixmap(QPixmap.fromImage(self.img))

    def clicked(self):
        self.webCamThread.initialize(self.img)
        self.webCamThread.start()

        
class WebCamThread(QThread):
    def __init__(self, lock, parent=None):
        super(WebCamThread, self).__init__(parent)
        self.lock = lock

    def initialize(self, img):
        self.img = img

    def run(self):
        cap = cv.CaptureFromCAM(0)
        while 1:
            capture = cv.QueryFrame(cap)
            if capture:
                pil = Image.fromstring("RGB", cv.GetSize(capture), capture.tostring(), "raw", "BGR")
                try:
                    self.lock.lockForWrite()
                    self.img = pil2qimage(pil, self.img)
                finally:
                    self.lock.unlock()

                self.emit(SIGNAL("update_pic()"))
            time.sleep(0.05)

win = WebCamDialog()
win.show()

sys.exit(app.exec_())