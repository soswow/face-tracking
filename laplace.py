from cvutils import  *

def laplace(img):
#    img = prepare_bw(img)
    img = normalize_plane(img, aggressive=0.005)
#    smooth = image_empty_clone(img)
    dst = cv.CreateImage(sizeOf(img), cv.IPL_DEPTH_16S, 1)
#    cv.Smooth(img, smooth, cv.CV_BILATERAL, 3, 3, 30,30)
#    cv.Laplace(img, dst, 3)
    cv.Sobel(img, dst, 0, 1, 3)
#    cv.
    cv.Convert(dst,img)
    return img

def main():
    img = cv.LoadImage("latex/Pictures/dr_house_gray.png", iscolor=False)
    cv.SaveImage("latex/Pictures/dr_house_gray_sobel.png", laplace(img))
#    with_webcam(laplace)

if __name__ == "__main__":
    main()
  