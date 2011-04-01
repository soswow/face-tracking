import cv
import numpy as np

def main():
    orig = cv.LoadImageM("sample/lena.bmp")
    np_mat = np.asarray(cv.GetMat(orig))
    np_mat = np.rot90(np_mat).copy()
    modif = cv.fromarray(np_mat)
    cv.SaveImage("sample/lena_90.png", modif)

if __name__ == '__main__':
    main()
