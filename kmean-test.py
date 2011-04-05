import cv

def main():
    #define MAX_CLUSTERS 5
#    CvScalar color_tab[MAX_CLUSTERS];

    img = cv.CreateImage( ( 500, 500 ), 8, 3 )
    rng = cv.RNG(500)

    color_tab = [cv.CV_RGB(255, 0, 0),
                 cv.CV_RGB(0, 255, 0),
                 cv.CV_RGB(100, 100, 255),
                 cv.CV_RGB(255, 0, 255),
                 cv.CV_RGB(255, 255, 0)]

    cv.NamedWindow( "clusters", 1 )

    while 1:
        cluster_count = cv.RandInt(rng)
        sample_count = cv.RandInt(rng)
        points = cv.CreateMat( sample_count, 1, cv.CV_32FC2 )
        clusters = cv.CreateMat( sample_count, 1, cv.CV_32SC1 )

        #/* generate random sample from multigaussian distribution */
        for k in range(cluster_count):
            center = [cv.RandInt(rng), cv.RandInt(rng)]
            point_chunk = cv.GetRows( points,
                       k*sample_count/cluster_count,
                       sample_count if k == (cluster_count - 1) else (k+1)*sample_count/cluster_count )
            cv.RandArr(rng, point_chunk, cv.CV_RAND_NORMAL,
                       (center[0],center[1],0,0),
                       (img.width/6, img.height/6,0,0))

#        /* shuffle samples */
#        for i in range(sample_count/2):
#            pt1 = points->data.fl + cvRandInt(&rng)
#            CvPoint2D32f* pt2 =
#                (CvPoint2D32f*)points->data.fl + cvRandInt(&rng)
#            cv.CV_SWAP( pt1, pt2, temp);
#        }

        cv.KMeans2( points, cluster_count, clusters,
                   ( cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1.0 ))

        cv.Zero(img)

        for i in range(sample_count):
            pt = points[i]
            cluster_idx = clusters[i]
            cv.Circle( img,
                      pt,
                      2,
                      color_tab[cluster_idx],
                      cv.CV_FILLED)

        cv.ShowImage( "clusters", img )

        key = cv.WaitKey(10)
        if key == 27:
            break

if __name__ == "__main__":
    main()