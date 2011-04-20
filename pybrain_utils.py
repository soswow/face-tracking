from __future__ import division

from pybrain.structure import SoftmaxLayer, SigmoidLayer, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.connections.identity import IdentityConnection
from pybrain.structure.modules.biasunit import BiasUnit
from pybrain.structure.modules.softmax import PartialSoftmaxLayer
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.supervised.trainers.rprop import RPropMinusTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import cv
import numpy as np
from cPickle import dump,load
import shutil

from utils import *
from sliding_window import *
from cvutils import *
from laplace import *

import math

root_folder = "/Users/soswow/Documents/Face Detection/test/"

def get_flatten_images_from(path,clazz):
    images = []
    for fullpath, name in directory_files(path):
        try:
            img = cv.LoadImage(fullpath,iscolor=False)
            images.append((clazz, get_flatten_image(img)))
        except IOError:
            pass
    return images

def get_nonfaces(set_n):
    path = root_folder + "sets/negative/%d" % set_n
    return get_flatten_images_from(path,0)

def get_faces(set_n):
    path = root_folder + "sets/positive/%d" % set_n
    return get_flatten_images_from(path,1)

def lena_test(net,net2):
#    img = scale_image(cv.LoadImage("sample/lena.bmp"))
    img = scale_image(cv.LoadImage("sample/Group-Oct06.jpg"))
    found = []
    
    buf_nf_sum= buf_f_sum=true_neg_sum=0
    k=0
    for i, (sample, box) in enumerate(samples_generator(img, 32,32,slide_step=4, resize_step=1.5,bw_from_v_plane=False)):
        nf, f = net.activate(get_flatten_image(sample))
        nf2, f2 = net2.activate(get_flatten_image(laplace(sample)))
        buf_nf, buf_f = tuple(net['out'].inputbuffer[0])
        buf_nf2, buf_f2 = tuple(net2['out'].inputbuffer[0])
        if f > nf and f2 > nf2 and buf_f > 250000 and buf_f2 > 50000:
            print "%d - %d %d" % (i, buf_f, buf_f2)
#            buf_nf, buf_f = tuple(net['out'].inputbuffer[0])
            buf_f_sum+=buf_f2
            buf_nf_sum+=buf_nf2
            found.append(box)
        else:
#            buf_nf, _ = tuple(net['out'].inputbuffer[0])
            true_neg_sum+=buf_nf2
            k+=1

    draw_boxes(found, img, color=cv.RGB(255,255,255), thickness=1, with_text=False)
    print "Avr nf %.3f, f %.3f, true negative %.3f" % (buf_nf_sum/len(found), buf_f_sum/len(found), true_neg_sum/k)
    show_image(img)

def build_exp_ann(indim, outdim):
    ann = FeedForwardNetwork()
    ann.addInputModule(LinearLayer(indim, name='in'))
    ann.addModule(SigmoidLayer(24, name='hidden'))
    ann.addOutputModule(SigmoidLayer(outdim,name='out'))
    ann.addModule(BiasUnit(name='bias'))

    hidd_neur_i = 0
    for cols, rows in ((1,8),(4,4)):#(2,2), (4,4),
        height = int(32/rows)
        width = int(32/cols)
        for col in range(cols):
            for row in range(rows):
                if width < 32:
                    for line in range(height):
                        fr = col*width + line*32 + row*32*height
                        to = fr + width

                        slices = {  'inSliceFrom': fr, 'inSliceTo': to,
                                    'outSliceFrom': hidd_neur_i, 'outSliceTo': hidd_neur_i+1,
                              'name': "%dx%d group (%dx%d) %d line Input(%d-%d) -> Hidden(%d-%d)" % (
                                cols, rows, col+1, row+1, line, fr, to, hidd_neur_i, hidd_neur_i+1)}

                        ann.addConnection(FullConnection(ann['in'], ann['hidden'], **slices))
                else:
                    fr = row * width * height
                    to = fr + width * height

                    slices = {  'inSliceFrom': fr, 'inSliceTo': to,
                                'outSliceFrom': hidd_neur_i, 'outSliceTo': hidd_neur_i+1,
                          'name': "%dx%d group (%dx%d) %d line Input(%d-%d) -> Hidden(%d-%d)" % (
                            cols, rows, col+1, row+1, 0, fr, to, hidd_neur_i, hidd_neur_i+1)}

                    ann.addConnection(FullConnection(ann['in'], ann['hidden'], **slices))
                hidd_neur_i+=1
    
    ann.addConnection(FullConnection(ann['hidden'], ann['out']))
    ann.addConnection(FullConnection(ann['bias'], ann['out']))
    ann.addConnection(FullConnection(ann['bias'], ann['hidden']))
    
    ann.sortModules()
    return ann


def build_ann(indim, outdim):
    ann = FeedForwardNetwork()

    ann.addInputModule(LinearLayer(indim, name='in'))
    ann.addModule(SigmoidLayer(5, name='hidden'))
#    ann.addModule(Normal(2,name='hidden'))
    ann.addOutputModule(SoftmaxLayer(outdim,name='out'))
    ann.addModule(BiasUnit(name='bias'))

    ann.addConnection(FullConnection(ann['in'], ann['hidden']))
    ann.addConnection(FullConnection(ann['hidden'], ann['out']))
#    ann.addConnection(FullConnection(ann['in'], ann['out']))
    ann.addConnection(FullConnection(ann['bias'], ann['out']))
    ann.addConnection(FullConnection(ann['bias'], ann['hidden']))

    ann.sortModules()

#    ann = buildNetwork(indim, outdim, outclass=SoftmaxLayer)
    return ann


def get_trained_ann(dataset, ann=None, test_train_prop=0.25, max_epochs=50):
    tstdata, trndata = dataset.splitWithProportion(test_train_prop)
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    if not ann:
        ann = build_ann(trndata.indim, trndata.outdim)
#        ann = build_exp_ann(trndata.indim, trndata.outdim)
#    trainer = RPropMinusTrainer(ann)
    trainer = BackpropTrainer(ann, dataset=trndata,learningrate=0.01, momentum=0.5, verbose=True)
    trnresult = tstresult = 0
#    for i in range(10):
    trainer.trainUntilConvergence(maxEpochs=max_epochs, verbose=True)
    trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
#        print trnresult, tstresult
    return ann, trnresult, tstresult

def dump_ann(ann, name="default-ann"):
    print "Saving ann"
    f = open(name, 'w')
    dump(ann, f)
    f.close()


def train_ann_with_dump(dataset, ann=None, name="default-ann"):
    ann, trnresult, tstresult =  get_trained_ann(dataset, ann=ann, max_epochs=30)
    dump_ann(ann, name)
    return trnresult, tstresult

def load_ann(name="default-ann"):
    print "Loading ann"
    f = open(name)
    ann = load(f)
    f.close()
    return ann

def extract_falses(ann1, ann2, check, testpath=None, falses_dir=None, do_copy=True,threshold=None):
    print "Testing ANN with all negatives or positives"
#    testpath = root_folder + "lenas"

    if do_copy:
        if os.path.exists(falses_dir):
            shutil.rmtree(falses_dir)
        else:
            os.makedirs(falses_dir)

    wrong = 0
    avg_level = 0
    avg_f=avg_p = 0
    tot=0
    prev_dir=None
    found_for_dir=0
    fk = 0
    found_list=[]
    avg_f_list=[]
    avg_p_list=[]
    def extract(found_list):
        for f, file in found_list:
            dest_name = os.path.join(falses_dir,"-".join(["%.2f" % (f*100)] + file.split(os.sep)[-2:]))
            if not os.path.exists(falses_dir):
                os.makedirs(falses_dir)
            shutil.copy(file, dest_name)
        found_list=[]
    
    for fullpath, name in yield_files_in_path(testpath):
        dir = fullpath.split(os.path.sep)[-2]
        try:
            if prev_dir and dir != prev_dir:
                print "In %s was wrong %d out of %d (%.2f%%)" % (prev_dir, found_for_dir,fk, found_for_dir/fk*100)
                found_for_dir=0
                fk = 0
                if do_copy:
                    extract(found_list)


            img = cv.LoadImage(fullpath, iscolor=False)
#            nf1, f1 = activate_with_threshold(ann,get_flatten_image(img),threshold=threshold)
            nf1, f1 = ann1.activate(get_flatten_image(img))
            nf2, f2 = ann2.activate(get_flatten_image(laplace(img)))
            buf_nf, buf_f = tuple(ann1['out'].inputbuffer[0])
#            diff = int(f2*100) - int(nf*100)
            value = buf_nf if nf1 > f1 else buf_f
            if check(nf1,f1) and check(nf2,f2):
                found_list.append((f1, fullpath))
                found_for_dir+=1
                wrong+=1
                avg_f_list.append(value)
                avg_f+= value
            else:
                a=0
                avg_p+= value
                avg_p_list.append(value)
            avg_level+=buf_nf if nf1 > f1 else buf_f
            fk+=1

            tot+=1
        except IOError:
            pass
        prev_dir = dir

    if found_list:
        extract(found_list)

    print "Error is %.2f%%" % (wrong/tot)
#    print "Found %d faces out of %d negatives (%.2f%%)" % (found, tot, found/tot*100)
    print "Average level (+) for all: %.2f" % (avg_level/tot)
    print "Average level (f) for trues: %.2f" % (avg_p/tot)
    print "Average level (nf) for falses: %.2f" % (avg_f/wrong)
    return wrong

#    maxx = max(avg_p_list)
#    def draw_hist(li):
#        si = len(li)
#        sqr = int(math.sqrt(si))
#        opa = sqr
#        for sq in range(sqr,si):
#            if not si % sq:
#                opa = sq
#                break
#        img = cv.GetImage(cv.fromarray(np.array(li).reshape(si/opa, opa)))
#
#        test = cv.CreateImage(sizeOf(img), cv.IPL_DEPTH_64F, 1)
#        pp = cv.CreateImage(sizeOf(img), cv.IPL_DEPTH_64F, 1)
#        cc = cv.CreateImage(sizeOf(img), 8, 1)
#        cv.Set(pp, maxx)
#        cv.Div(img,pp,test,scale=255)
#        cv.Convert(test, cc)
#
#        hist = cv.CreateHist([800], cv.CV_HIST_ARRAY, [(0,255)], 1)
#        cv.CalcHist([cc], hist)
#        hist_img = get_hist_image(hist, 800, width=800)
#        show_image(hist_img)
#
#    draw_hist(avg_p_list)
#    draw_hist(avg_f_list)

def train_and_save_ann(ann=None, set_n=1, non_face_prop=2, faces_num=1000):
    alldata = ClassificationDataSet(32 * 32, 1, nb_classes=2, class_labels=("Non-Face", "Face"))
    print "Loading data"
    nonfaces = get_nonfaces(set_n)
    random.shuffle(nonfaces)
    faces = get_faces(set_n)[:faces_num]
    random.shuffle(faces)

    nonfaces_num = int(len(faces) * non_face_prop)
    all_samples = nonfaces[:nonfaces_num] + faces
    print "Training on %d samples (%d faces vs %s non-faces)" % (len(all_samples), len(faces), nonfaces_num)

    for clazz, sample in all_samples:
        alldata.addSample(sample, clazz)
    print "Training ann"
    trnresult, tstresult = train_ann_with_dump(alldata,ann=ann)
    #    ann = load_ann()
    print "  train error: %5.2f%%" % trnresult,\
    "  test error: %5.2f%%" % tstresult
    return tstresult

def train_on_falses(ann=None, set_n=1):
    print "Training on found false negatives and positives"

    alldata = ClassificationDataSet(32 * 32, 1, nb_classes=2, class_labels=("Non-Face", "Face"))

    for clazz, listt in ((1, directory_files(root_folder + "false_negatives/")),
        (0, directory_files(root_folder + "false_positives/")), ):

    #    random.shuffle(listt)
        for filename, name in listt:
            try:
                sample = get_flatten_image(cv.LoadImage(filename, iscolor=False))
            except IOError:
                continue
            alldata.addSample(sample, clazz)

    all_samples = get_nonfaces(set_n) + get_faces(set_n)
    for clazz, sample in all_samples:
        alldata.addSample(sample, clazz)

    trnresult, tstresult = train_ann_with_dump(alldata,ann=ann)
    #    ann = load_ann()
    print "  train error: %5.2f%%" % trnresult,\
    "  test error: %5.2f%%" % tstresult

def activate_with_threshold(ann, sample, threshold):
    nf, f = ann.activate(sample)
    buf_nf, buf_f = tuple(ann['out'].inputbuffer[0])
    if not threshold:
        return nf, f

    if (nf > f and buf_nf > threshold) or (f > nf and buf_f > threshold):
        return nf, f
    else:
        return f, nf

def test_ann_on_set(ann, set_n=2, threshold=None):
    print "Testing on set %d" % set_n
    all_samples = get_nonfaces(set_n) + get_faces(set_n)
    trues = 0
    false_positives = 0
    false_negatives = 0
    falses = 0
    random.shuffle(all_samples)
    for clazz, sample in all_samples:
        nf, f = activate_with_threshold(ann,sample,threshold)
        if clazz and f > nf or not clazz and nf > f:
            trues+=1
        elif clazz and nf > f:
            false_negatives +=1
        elif not clazz and f > nf:
            false_positives+=1
        else:
            falses+=1

    total = len(all_samples)
    print "Total tested: %d " % total
    print "Trues: %d (%.2f%%)" % (trues, (trues/total*100))
    false_negative_perc = (false_negatives / total * 100)
    print "False negatives: %d (%.2f%%)" % (false_negatives, false_negative_perc)
    false_positive_perc = (false_positives / total * 100)
    print "False positives: %d (%.2f%%)" % (false_positives, false_positive_perc)
    if falses > 0:
        print "And some stranfe falses: %d" % falses
    return false_negative_perc, false_positive_perc

def main():
    global root_folder

#    ann = build_exp_ann(1024, 2)
    for i in range(2):
        train_and_save_ann(set_n=1, non_face_prop=2, faces_num=1500)
    #    ann_pixel = load_ann('default-ann7.9')
    #    ann_edge = load_ann()
        ann = load_ann()
        test_ann_on_set(ann)
#    test_all_negatives(ann)
    print "sobel!"
    root_folder += "sobel/"
#    ann = build_exp_ann(1024, 2)
    for i in range(2):
        train_and_save_ann(set_n=1, non_face_prop=2, faces_num=1500)
    #    ann_pixel = load_ann('default-ann7.9')
    #    ann_edge = load_ann()
        ann = load_ann()
        test_ann_on_set(ann)

#    lena_test(ann_pixel, ann_edge)

    

    #------
#    totals = []
#    for threshold in range(0, 15000, 1000):
#        print "\nThreshold: %d" % threshold
#    threshold=None
#    avg_errors = {"total":0, "fp":0, "fn":0 }
#    rng=range(1,5)
#    for k in rng:
#        false_negative_perc, false_positive_perc = test_ann_on_set(ann, set_n=k, threshold=threshold)
#        avg_errors["fp"]+=false_positive_perc
#        avg_errors["fn"]+=false_negative_perc
#        avg_errors["total"]+=false_negative_perc+false_positive_perc
##
#    print "Total false positive: %.2f" % (avg_errors["fp"] / len(rng))
#    print "Total false negative: %.2f" % (avg_errors["fn"] / len(rng))
#    print "Total avg error: %.2f" % (avg_errors["total"] / len(rng))
#        totals.append((threshold, avg_errors))

#    print "\n\n!!!!"
#    for threshold, avg_errors in totals:
#        print "Threshold: %d -> fp: %.2f, fn: %.2f, total: %.2f" % (
#        threshold, avg_errors["fp"], avg_errors["fn"], avg_errors["total"])
    #------
#
#    print "ann 7.9  "
#    ann = load_ann('default-ann7.9')
#    test_ann_on_set(ann)
#
#    for k in range(1,5):
#        test_ann_on_set(ann, set_n=k)

#    print "Extracting false positives"
#    extract_falses(ann_pixel, ann_edge, (lambda fn, f:f > fn), root_folder + "negative",
#                   root_folder + "false_positives", do_copy=False)
###
#    print "Extracting false_positives negatives"
#    extract_falses(ann_pixel, ann_edge, (lambda fn, f:f < fn), root_folder + "positive",
#                   root_folder + "false_negatives", do_copy=False)

#    train_on_falses()
#    ann = load_ann()
#    test_ann_on_set(ann)

    
#    arr = get_flatten_image(cv.LoadImage("sample/chernobyl.png", iscolor=False))


#    nf, f = ann.activate(arr)
#    buf_nf, buf_f = tuple(ann['out'].inputbuffer[0])
#    print nf,f,buf_nf,buf_f
#
#    arr = get_flatten_image(cv.LoadImage("sample/s5-3.png", iscolor=False))
#    nf, f = ann.activate(arr)
#    buf_nf, buf_f = tuple(ann['out'].inputbuffer[0])
#    print nf,f,buf_nf,buf_f

#    test_all_negatives(ann)

#    path = root_folder + "sets/positive/1/"
#    path = root_folder + "sets/negative/2/"

#    path = root_folder + "webcam"
#    listt = directory_files(path)
#    p=0
#    for filename, name in listt:
#        try:
#            arr = get_flatten_image(cv.LoadImage(filename, iscolor=False))
#        except IOError:
#            continue
#        nf, f = ann.activate(arr)
#        buf_nf, buf_f = tuple(ann['out'].inputbuffer[0])
#        if f > nf:
#            p+=1
#            print nf,f,buf_nf,buf_f,name
#        else:
#            print ""
#    print p

#    test_all_negatives(ann)


if __name__ == "__main__":
    main()