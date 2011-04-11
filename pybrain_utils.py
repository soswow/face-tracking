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

import math

root_folder = "/Users/soswow/Documents/Face Detection/test/"

def get_flatten_image(img):
    arr = np.asarray(cv.GetMat(img))
    return arr.flatten()

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

def lena_test(net):
#    img = cv.LoadImage("sample/lena.bmp")
    img = scale_image(cv.LoadImage("sample/Group-Oct06.jpg"))
    found = []
    b=[32000,32200]
    buf_nf_sum= buf_f_sum=true_neg_sum=0
    k=0
    for i, (sample, box) in enumerate(samples_generator(img, 32,32,slide_step=2, resize_step=1.5,bw_from_v_plane=False)):
        nf, f = net.activate(get_flatten_image(sample))
        buf_nf, buf_f = tuple(net['out'].inputbuffer[0])
        if f > nf and buf_f > 325000:
            print "%d - %d %d" % (i, buf_nf, buf_f)
#            buf_nf, buf_f = tuple(net['out'].inputbuffer[0])
            buf_f_sum+=buf_f
            buf_nf_sum+=buf_nf
            found.append(box)
        else:
#            buf_nf, _ = tuple(net['out'].inputbuffer[0])
            true_neg_sum+=buf_nf
        k+=1

    draw_boxes(found, img, color=cv.RGB(255,255,255), thickness=1, with_text=False)
    print "Avr nf %.3f, f %.3f, true negative %.3f" % (buf_nf_sum/len(found), buf_f_sum/len(found), true_neg_sum/k)
    show_image(img)


def build_ann(indim, outdim):
    ann = FeedForwardNetwork()

    ann.addInputModule(LinearLayer(indim, name='in'))
#    ann.addModule(SigmoidLayer(4, name='hidden'))
#    ann.addModule(Normal(2,name='hidden'))
    ann.addOutputModule(SoftmaxLayer(outdim,name='out'))
    ann.addModule(BiasUnit(name='bias'))

#    ann.addConnection(FullConnection(ann['in'], ann['hidden']))
#    ann.addConnection(FullConnection(ann['hidden'], ann['out']))
    ann.addConnection(FullConnection(ann['in'], ann['out']))
    ann.addConnection(FullConnection(ann['bias'], ann['out']))
#    ann.addConnection(FullConnection(ann['bias'], ann['hidden']))
    
    ann.sortModules()

#    ann = buildNetwork(indim, outdim, outclass=SoftmaxLayer)
    return ann


def get_trained_ann(dataset, ann=None, test_train_prop=0.25, max_epochs=50):
    tstdata, trndata = dataset.splitWithProportion(test_train_prop)
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    if not ann:
        ann = build_ann(trndata.indim, trndata.outdim)
#    trainer = RPropMinusTrainer(ann)
    trainer = BackpropTrainer(ann, dataset=trndata, verbose=True) #learningrate=0.1, momentum=0.1
    trainer.trainUntilConvergence(maxEpochs=max_epochs, verbose=True)
    trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
    return ann, trnresult, tstresult

def dump_ann(ann, name="default-ann"):
    print "Saving ann"
    f = open(name, 'w')
    dump(ann, f)
    f.close()


def train_ann_with_dump(dataset, ann=None, name="default-ann"):
    ann, trnresult, tstresult =  get_trained_ann(dataset, ann=ann, max_epochs=50)
    dump_ann(ann, name)
    return trnresult, tstresult

def load_ann(name="default-ann"):
    print "Loading ann"
    f = open(name)
    ann = load(f)
    f.close()
    return ann

def extract_falses(ann, check, testpath=None, falses_dir=None):
    print "Testing ANN with all negatives or positives"
#    testpath = root_folder + "lenas"

    if os.path.exists(falses_dir):
        shutil.rmtree(falses_dir)
    else:
        os.makedirs(falses_dir)

    wrong = 0
#    avg_level = 0
    avg_f = 0
    tot=0
    prev_dir=None
    found_for_dir=0
    fk = 0
    found_list=[]

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
                extract(found_list)


            img = cv.LoadImage(fullpath, iscolor=False)
            nf, f = ann.activate(get_flatten_image(img))
#            buf_nf, buf_f = ann['out'].inputbuffer
#            diff = int(f*100) - int(nf*100)
            if check(nf,f):
                found_list.append((f, fullpath))
                found_for_dir+=1
                wrong+=1
#                avg_f+=f
            fk+=1
#            avg_level+=nf
            tot+=1
        except IOError:
            pass
        prev_dir = dir

    if found_list:
        extract(found_list)

    print "Error is %.2f%%" % (wrong/tot*100)
#    print "Found %d faces out of %d negatives (%.2f%%)" % (found, tot, found/tot*100)
#    print "Average NotFound for all: %.2f" % (avg_level/tot*100)
#    print "Average Found for found cases: %.2f" % (avg_f/tot*100)


def train_and_save_ann(ann=None, set_n=1):
    alldata = ClassificationDataSet(32 * 32, 1, nb_classes=2, class_labels=("Non-Face", "Face"))
    print "Loading data"
    all_samples = get_nonfaces(set_n) + get_faces(set_n)
    random.shuffle(all_samples)
    for clazz, sample in all_samples[:2000]:
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

    for clazz, listt in (#(1, directory_files(root_folder + "false_negatives/")),
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

def test_ann_on_set(ann, set_n=2):
    print "Testing on set %d" % set_n
    all_samples = get_nonfaces(set_n) + get_faces(set_n)
    trues = 0
    false_positives = 0
    false_negatives = 0
    falses = 0
    for clazz, sample in all_samples:
        nf, f = ann.activate(sample)
        if clazz and f or not clazz and nf:
            trues+=1
        elif clazz and nf:
            false_negatives +=1
        elif not clazz and f:
            false_positives+=1
        else:
            falses+=1

    total = len(all_samples)
    print "Total tested: %d " % total
    print "Trues: %d (%.2f%%)" % (trues, (trues/total*100))
    print "False negatives: %d (%.2f%%)" % (false_negatives, (false_negatives/total*100))
    print "False positives: %d (%.2f%%)" % (false_positives, (false_positives/total*100))
    if falses > 0:
        print "And some stranfe falses: %d" % falses

def main():
#    global root_folder
#    root_folder += "with_mask/"
    train_and_save_ann()
    ann = load_ann()
    test_ann_on_set(ann)
#    test_all_negatives(ann)
    for k in range(1,5):
        test_ann_on_set(ann, set_n=k)

#
#    print "ann 7.9  "
#    ann = load_ann('default-ann7.9')
#    test_ann_on_set(ann)
#
#    for k in range(1,5):
#        test_ann_on_set(ann, set_n=k)

#    print "Extracting false positives"
#    extract_falses(ann, (lambda fn, f:f > fn), root_folder + "negative",
#                   root_folder + "false_positives")
#    print "Extracting false negatives"
#    extract_falses(ann, (lambda fn, f:f < fn), root_folder + "positive",
#                   root_folder + "false_negatives")
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