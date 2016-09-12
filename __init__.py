from __future__ import print_function
import numpy
import gzip
import pickle# as pickle
from collections import OrderedDict

import os
#os.environ["THEANO_FLAGS"]="optimizer_excluding=local_gpu_advanced_incsubtensor1"


import warnings
import inspect

for i in inspect.stack():
    fi = inspect.getframeinfo(i[0])
    if fi.code_context is None:
        continue
    for j in fi.code_context:
        if (("from theano_tools import*" in j) or
            ("from theano_tools import *" in j) or
            ("import theano_tools\n" in j) or
            ("import theano_tools " in j)):
            warnings.warn("top level import of theano_tools may not import what you expect!")

import util

full_ls = util.ls


def nponehot(x, n):
    a = numpy.zeros(n, 'float32')
    a[x] = 1
    return a

def savepkl(x,name):
    pickle.dump(x, open(name,'wb'),-1)

def loadpkl(name):
    return pickle.load(open(name,'rb'))

def savepklgz(x,name,addExt=True):
    pickle.dump(x, gzip.open(name+('.pkl.gz' if addExt else ''),'wb'),-1)

def loadpklgz(name,addExt=True):
    return pickle.load(gzip.open(name+('.pkl.gz' if addExt else ''),'rb'))


def load_dotpaths():
    """looks for dataset paths in ~/.datasetpaths so that code ran on a cluster doesn't have
    to be modified each time

    """
    import os
    home = os.environ["HOME"]
    if os.path.exists(home+"/.datasetpaths"):
        paths = dict(map(lambda x:x.split(":"), open(home+"/.datasetpaths",'r').read().splitlines()))
        return paths
    return None

class GenericClassificationDataset:
    paths = load_dotpaths()
    def __init__(self, which, alt_path=None, doDiv255=False, _pass=False):
        self.alt_path = alt_path
        self.doDiv255 = doDiv255
        if self.doDiv255:
            self.pp = lambda x,y:(numpy.float32(x/255.), y)
        else:
            self.pp = lambda x,y:(x,y)

        if _pass: return
        if which == "mnist":
            self.load_mnist()
        elif which == "cifar10":
            self.load_cifar10()
        elif which == "svhn":
            self.load_svhn()
        elif which == "covertype":
            self.load_covertype()
        elif which == 'custom':
            self.load_custom()
        else:
            raise ValueError("Don't know about this dataset: '%s'"%which)

    def load_custom(self):
        p = self.alt_path
        self.train,self.valid,self.test = pickle.load(gzip.open(p,'r'))


    def makeDatasetForClasses(self, cs, doCorrectYs=True):
        cs = list(cs)
        if doCorrectYs:
            # if cs is e.g. [4,6,9], the new ys
            # will need to be corrected to [0,1,2]:
            def cr(y):
                b = numpy.zeros(max(cs)+1,dtype='int32')
                b[cs] = range(len(cs))
                return b[y]
        else:
            cr = lambda x:x
        new = GenericClassificationDataset("",_pass=True)

        newTrain = zip(*[(x,y) for x,y in zip(*self.train)
                         if y in cs])
        new.train = [numpy.float32(newTrain[0]),
                     cr(numpy.int32(newTrain[1]))]

        newValid = zip(*[(x,y) for x,y in zip(*self.valid)
                         if y in cs])
        new.valid = [numpy.float32(newValid[0]),
                     cr(numpy.int32(newValid[1]))]

        newTest = zip(*[(x,y) for x,y in zip(*self.test)
                         if y in cs])
        new.test = [numpy.float32(newTest[0]),
                    cr(numpy.int32(newTest[1]))]
        return new

    def augmentTrain(self, method, imageShape=(3,32,32)):
        if method == 'horizontal flip':
            X = self.train[0]
            X = X.reshape([-1]+list(imageShape))


    def load_covertype(self):
        f = gzip.open(self.alt_path if self.alt_path else "/data/UCI/covtype.pkl.gz", 'rb')
        numpy.random.seed(142857)
        data = pickle.load(f)
        X = data[:,:-1]
        Y = data[:,-1] - 1 # 1-based labels -> 0-based
        
        # these numbers are from the dataset instructions,
        # they incur a perfect balance of the training classes in the dataset
        # but then the test set is 50x bigger than the training set...?
        # From looking at the data:
        #   Y = numpy.int32(Y)
        #   print(numpy.bincount(Y[:11340]))
        #   print(numpy.bincount(Y[11340:11340+3780]))
        #   print(numpy.bincount(Y[11340+3780:]))
        # you can see that one of the classes has very few examples,
        # so to keep balanced classes in training, you can only take
        # so many examples from that class
        ntrain = 11340
        nvalid = 3780
        self.train = [X[:ntrain], Y[:ntrain]]
        self.valid = [X[ntrain:ntrain+nvalid],
                      Y[ntrain:ntrain+nvalid]]
        self.test = [X[ntrain+nvalid:],
                     Y[ntrain+nvalid:]]
        
        self.input_shape = X.shape[1]
        if 0:
            mu = self.train[0].mean(axis=0)
            sigma = self.train[0].std(axis=0)
            self.train[0] = (self.train[0]-mu) / sigma
            self.valid[0] = (self.valid[0]-mu) / sigma
            self.test[0]  = (self.test[0] -mu) / sigma
        else:
            min = self.train[0].min(axis=0)
            max = self.train[0].max(axis=0)
            max[max==0] = 1
            self.train[0] = (self.train[0] - min) / max
            self.valid[0] = (self.valid[0] - min) / max
            self.test[0] = (self.test[0] - min) / max
            
    def load_mnist(self):
        path = self.alt_path if self.alt_path else "mnist.pkl.gz"
        if not os.path.exists(path):
            if GenericClassificationDataset.paths is None:
                raise Exception("given path doesn't exist, try to define it in ~/.datasetpaths")
            path = GenericClassificationDataset.paths['mnist']

        f = gzip.open(path, 'rb')
        #self.train,self.valid,self.test = map(list,pickle.load(f,encoding='latin1'))
        self.train,self.valid,self.test = map(list,pickle.load(f))
        f.close()

        #self.train[0] = self.train[0]/self.train[0].std(axis=0) - self.train[0].mean()
        #self.valid[0] = self.valid[0]/self.valid[0].std(axis=0) - self.valid[0].mean()
        #self.test[0] = self.test[0]/self.test[0].std(axis=0) - self.test[0].mean()
        self.train[1] = numpy.uint8(self.train[1])
        self.valid[1] = numpy.uint8(self.valid[1])
        self.test[1] = numpy.uint8(self.test[1])

        self.ntrain = 1.*self.train[0].shape[0]
        self.nvalid = 1.*self.valid[0].shape[0]
        self.ntest = 1.*self.test[0].shape[0]

    def load_cifar10(self):
        path = self.alt_path if self.alt_path else 'cifar_10_shuffled.pkl'
        if not os.path.exists(path):
            if GenericClassificationDataset.paths is None:
                raise Exception("given path doesn't exist, try to define it in ~/.datasetpaths")
            path = GenericClassificationDataset.paths['cifar10']

        print(path)

        trainX, trainY, testX, testY = pickle.load(open(path,'rb'))
        trainX = numpy.float32(trainX / 255.)
        testX = numpy.float32(testX / 255.)
        assert trainX.shape == (50000, 3, 32, 32)
        print(testX.shape, trainX.shape)
        print(testX.mean(),trainX.mean())
        self.train = [trainX[:40000], trainY[:40000]]
        self.valid = [trainX[40000:], trainY[40000:]]
        self.test = [testX, testY]
        self.ntrain = 1.*self.train[0].shape[0]
        self.nvalid = 1.*self.valid[0].shape[0]
        self.ntest = 1.*self.test[0].shape[0]

    def load_svhn(self):
        train, test = pickle.load(open(self.alt_path,'rb'))
        train[1] = train[1].flatten() - 1
        test[1] = test[1].flatten() - 1
        n = 580000
        self.train = [train[0][:n], train[1][:n]]
        self.valid = [train[0][n:], train[1][n:]]
        self.test = test

    def trainIndices(self, minibatch_size=32):
        nminibatches = self.train[0].shape[0] / minibatch_size
        indexes = numpy.arange(nminibatches)
        numpy.random.shuffle(indexes)
        for i in indexes:
            yield i

    def trainMinibatches(self, minibatch_size=32):
        nminibatches = self.train[0].shape[0] / minibatch_size
        indexes = numpy.arange(nminibatches)
        numpy.random.shuffle(indexes)
        for i in indexes:
            yield self.pp(self.train[0][i*minibatch_size:(i+1)*minibatch_size],
                          self.train[1][i*minibatch_size:(i+1)*minibatch_size])

    def validMinibatches(self, minibatch_size=32):
        nminibatches = self.valid[0].shape[0] / minibatch_size
        indexes = numpy.arange(nminibatches)
        numpy.random.shuffle(indexes)
        for i in indexes:
            yield self.pp(self.valid[0][i*minibatch_size:(i+1)*minibatch_size],
                          self.valid[1][i*minibatch_size:(i+1)*minibatch_size])

    def testMinibatches(self, minibatch_size=32):
        nminibatches = self.test[0].shape[0] / minibatch_size
        indexes = numpy.arange(nminibatches)
        numpy.random.shuffle(indexes)
        for i in indexes:
            yield self.pp(self.test[0][i*minibatch_size:(i+1)*minibatch_size],
                          self.test[1][i*minibatch_size:(i+1)*minibatch_size])


    def validate(self, test, minibatch_size=32):
        cost = 0.0
        error = 0.0
        for x,y in self.validMinibatches(minibatch_size):
            c,e = test(x,y)
            cost += c
            error += e
        return (error / self.valid[0].shape[0],
                cost /  self.valid[0].shape[0])

    def doTest(self, test, minibatch_size=32):
        cost = 0.0
        error = 0.0
        for x,y in self.testMinibatches(minibatch_size):
            c,e = test(x,y)
            cost += c
            error += e
        return (error / self.test[0].shape[0],
                cost /  self.test[0].shape[0])

def get_pseudo_srqt(x):
    sqrtx = numpy.sqrt(x)
    miny = 1
    for i in range(2,x/2+1):
        if x % i == 0:
            if (sqrtx-i)**2 < (sqrtx-miny)**2:
                miny = i
    return miny


class GeneratingDataset(GenericClassificationDataset):
    def __init__(self, generator):
        self.generator = generator
    def trainMinibatches(self, minibatch_size=32, n=1000):
        for i in range(n):
            d = numpy.float32([self.generator() for i in range(minibatch_size)])
            yield (d,0)

class tools:
    @staticmethod
    def export_feature_image(w, path, img_shape):
        if isinstance(w, T.sharedvar.SharedVariable):
            w = w.get_value()
        import scipy.misc as misc
        w = w.T
        ps = get_pseudo_srqt(w.shape[0])
        if len(img_shape) == 2:
            w = w.reshape([ps, w.shape[0]/ps, img_shape[0], img_shape[1]])
        elif len(img_shape) == 3:
            w = w.reshape([ps, w.shape[0]/ps, img_shape[0], img_shape[1], img_shape[2]])
        else:
            raise ValueError(img_shape)

        misc.imsave(path, numpy.hstack(numpy.hstack(w)))

    @staticmethod
    def export_many_images(images, path):
        import scipy.misc as misc
        w = images
        ps = get_pseudo_srqt(w.shape[0])
        img_shape = w.shape[1:]
        if len(img_shape) == 2:
            w = w.reshape([ps, w.shape[0]/ps, img_shape[0], img_shape[1]])
        elif len(img_shape) == 3:
            w = w.reshape([ps, w.shape[0]/ps, img_shape[0], img_shape[1], img_shape[2]])
        else:
            raise ValueError(img_shape)

        misc.imsave(path, numpy.hstack(numpy.hstack(w)))

    @staticmethod
    def export_simple_plot1d(ys,path,ylabel="",xlabel="",format=None, color=(0,0,1,1)):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        pyplot.clf()
        pyplot.plot(numpy.arange(len(ys)), ys, color=color)
        pyplot.grid(True)
        pyplot.show(block=False)
        if ylabel: pyplot.ylabel(ylabel)
        if xlabel: pyplot.xlabel(xlabel)
        pyplot.savefig(path, format=format)

    @staticmethod
    def export_multi_plot1d(ys,path,ylabel="",labels=[],format=None):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        pyplot.clf()
        if len(labels) < len(ys):
            labels += [""]*(len(ys)-len(labels))
        for i,l in zip(ys,labels):
            pyplot.plot(numpy.arange(len(i)), i, label=l)
        if len(labels):
            pyplot.legend()
        pyplot.show(block=False)
        pyplot.grid(True)
        if ylabel: pyplot.ylabel(ylabel)
        pyplot.savefig(path, format=format)

    @staticmethod
    def open_video(path, method='avconv', outputsize='800x600', fps=60):
        from subprocess import Popen, PIPE
        video = Popen([method, '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg',
                       '-r', str(fps),'-s',outputsize, '-i', '-',
                       '-qscale', '9', '-r', str(fps), path], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        return video
