import numpy
import scipy.misc
import gzip
import cPickle# as pickle
from collections import OrderedDict


import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .sparse_dot import*
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool



def savepkl(x,name):
    cPickle.dump(x, file(name,'w'),-1)

def loadpkl(name):
    return cPickle.load(file(name,'r'))

def momentum(epsilon):
    def gradient_descent(params, grads, lr):
        mom_ws = [theano.shared(0*(i.get_value()+1), i.name+" momentum")
                  for i in params]
        mom_up = [(i, epsilon * i + (1-epsilon) * gi)
                  for i,gi in zip(mom_ws, grads)]
        up = [(i, i - lr * mi) for i,mi in zip(params, mom_ws)]
        return up+mom_up
    return gradient_descent


def gradient_descent(params, grads, lr):
    up = [(i, i - lr * gi) for i,gi in zip(params, grads)]
    return up


def adagrad(params, grads, lr):
    hist = [theano.shared(0*(i.get_value()+1), i.name+" hist")
            for i in params]
    hist_up = [(i, i+gi**2)
               for i,gi in zip(hist, grads)]
    up = [(i, i - lr * gi / T.sqrt(1e-6 + hi + gi**2)) for i,gi,hi in zip(params,grads, hist)]
    return up+hist_up

def rmsprop(decay, epsilon=1e-3):
    def sgd(params, grads, lr):
        updates = []
        # shared variables
        mean_square_grads = [theano.shared(i.get_value()*0.+1) for i in params]
        # msg updates:
        new_mean_square_grads = [decay * i + (1 - decay) * T.sqr(gi)
                                 for i,gi in zip(mean_square_grads, grads)]
        updates += [(i,ni) for i,ni in zip(mean_square_grads,new_mean_square_grads)]
        # cap sqrt(i) at epsilon
        rms_grad_t = [T.maximum(T.sqrt(i), epsilon) for i in new_mean_square_grads]
        # actual updates
        delta_x_t = [lr * gi / rmsi for gi,rmsi in zip(grads, rms_grad_t)]
        updates += [(i, i-delta_i)
                    for i,delta_i in zip(params,delta_x_t)]
        return updates
    return sgd


class adam:
    def __init__(self,
                 beta1 = 0.9, beta2 = 0.999, epsilon = 1e-4):
        self.b1 = numpy.float32(beta1)
        self.b2 = numpy.float32(beta2)
        self.eps = numpy.float32(epsilon)

    def __call__(self, params, grads, lr):
        t = theano.shared(numpy.array(2., dtype = 'float32'))
        updates = OrderedDict()
        updates[t] = t + 1

        for param, grad in zip(params, grads):
            last_1_moment = theano.shared(numpy.float32(param.get_value() * 0))
            last_2_moment = theano.shared(numpy.float32(param.get_value() * 0))

            new_last_1_moment = T.cast((1. - self.b1) * grad + self.b1 * last_1_moment, 'float32')
            new_last_2_moment = T.cast((1. - self.b2) * grad**2 + self.b2 * last_2_moment, 'float32')

            updates[last_1_moment] = new_last_1_moment
            updates[last_2_moment] = new_last_2_moment
            updates[param] = (param - (lr * (new_last_1_moment / (1 - self.b1**t)) /
                                      (T.sqrt(new_last_2_moment / (1 - self.b2**t)) + self.eps)))

        return updates.items()


class SharedGenerator:
    def __init__(self):
        self.reset()
    def reset(self):
        self.param_list = [] # currently bound list of parameters
        self.param_groups = {} # groups of parameters
        self.param_costs = {} # each group can have attached costs
    def bind(self, params, name="default"):
        if type(params)==str:
            self.param_list = self.param_groups[params]
            return
        self.param_list = params
        self.param_groups[name] = params
        if name not in self.param_costs:
            self.param_costs[name] = []
    def __call__(self, name, shape, init='uniform', **kwargs):
        #print "init",name,shape,init,kwargs
        if init == "uniform" or init == "glorot":
            k = numpy.sqrt(6./numpy.sum(shape)) if 'k' not in kwargs else kwargs['k']
            values = numpy.random.uniform(-k,k,shape)
        elif init == "bengio":
            p = kwargs['inputDropout'] if 'inputDropout' in kwargs and kwargs['inputDropout'] else 1
            k = numpy.sqrt(6.*p/shape[0]) if 'k' not in kwargs else kwargs['k']
            values = numpy.random.uniform(-k,k,shape)
        elif init == "one":
            values = numpy.ones(shape)
        elif init == "zero":
            values = numpy.zeros(shape)
        elif type(init) == type(numpy.ndarray):
            values = init
        else:
            raise ValueError(init)
        s = theano.shared(numpy.float32(values), name=name)
        self.param_list.append(s)
        return s

    def exportToFile(self, path):
        exp = {}
        for g in self.param_groups:
            exp[g] = [i.get_value() for i in self.param_groups[g]]
        cPickle.dump(exp, file(path,'w'), -1)

    def importFromFile(self, path):
        exp = cPickle.load(file(path,'r'))
        for g in exp:
            for i in range(len(exp[g])):
                print g, exp[g][i].shape
                self.param_groups[g][i].set_value(exp[g][i])
    def attach_cost(self, name, cost):
        self.param_costs[name].append(cost)
    def get_costs(self, name):
        return self.param_costs[name]
    def get_all_costs(self):
        return [j  for i in self.param_costs for j in self.param_costs[i]]
    def get_all_names(self):
        print [i for i in self.param_costs]
        print self.param_costs.keys()
        return self.param_costs.keys()

    def computeUpdates(self, lr, gradient_method=gradient_descent):
        updates = []
        for i in self.param_costs:
            updates += self.computeUpdatesFor(i, lr, gradient_method)
        return updates

    def computeUpdatesFor(self, name, lr, gradient_method=gradient_descent):
        if name not in self.param_costs or \
           not len(self.param_costs[name]):
            return []
        cost = sum(self.param_costs[name])
        grads = T.grad(cost, self.param_groups[name])
        updates = gradient_method(self.param_groups[name], grads, lr)

        return updates


shared = SharedGenerator()


relu = lambda x: T.maximum(0,x)


class ConvBatchNormalization:
    def __init__(self, n):
        self.n = n
        self.gamma = shared('gamma', (n,), "one")
        self.beta = shared('beta', (n,), "zero")
    def __call__(self, x, *args):

        means = T.mean(x, axis=[0,2,3]).dimshuffle('x',0,'x','x')
        std = (T.std(x, axis=[0,2,3]) + 1e-5).dimshuffle('x',0,'x','x')
        out = (x - means) / std
        out = self.gamma.dimshuffle('x',0,'x','x') * out + self.beta.dimshuffle('x',0,'x','x')
        return out

class HiddenLayer:
    def __init__(self, n_in, n_out, activation, init="bengio", canReconstruct=False,
                 inputDropout=None,name=""):
        self.W = shared("W"+name, (n_in, n_out), init, inputDropout=inputDropout)
        self.b = shared("b"+name, (n_out,), "zero")
        self.activation = activation
        self.params = [self.W, self.b]
        if canReconstruct:
            self.bprime = shared("b'"+name, (n_in,), "zero")
            self.params+= [self.bprime]
        self.name=name
    def orthonormalize(self):
        import numpy.linalg
        from scipy.linalg import sqrtm, inv

        def sym(w):
            return w.dot(inv(sqrtm(w.T.dot(w))))
        Wval = numpy.random.normal(0,1,self.W.get_value().shape)
        Wval = sym(Wval).real
        self.W.set_value(numpy.float32(Wval))

    def __call__(self, x, *args):
        return self.activation(T.dot(x,self.W) + self.b)
    def reconstruct(self, x):
        return self.activation(T.dot(x,self.W.T) + self.bprime)

class InputSparseHiddenLayer:
    def __init__(self, n_in, n_out, activation, init="uniform", block_size=None):
        self.W = shared("W", (n_in, n_out), init)
        self.b = shared("b", (n_out,), "zero")
        self.activation = activation
        assert block_size != None
        self.block_size = block_size

    def __call__(self, x, xmask):
        print xmask
        return self.activation(sparse_dot(x, xmask, self.W, None, self.b, self.block_size))



class PolicyDropoutLayer:
    def __init__(self, n_in, n_out, block_size, activation, do_dropout=False,
                 reinforce_params="reinforce",
                 default_params="default"):
        self.block_size = block_size
        self.nblocks = n_out / block_size
        self.do_dropout = do_dropout
        assert n_out % block_size == 0

        self.h = HiddenLayer(n_in, n_out, activation)
        shared.bind(reinforce_params)
        self.d = HiddenLayer(n_in, self.nblocks, T.nnet.sigmoid)
        shared.bind(default_params)

    def __call__(self, x, xmask=None):
        probs = self.d(x) * 0.98 + 0.01 # necessary epsilon for numerical reasons
        mask = srng.uniform(probs.shape) < probs
        mask.name = "mask!"
        masked = self.h.activation(sparse_dot(x, xmask, self.h.W, mask, self.h.b, self.block_size))
        
        if not "this is the equivalent computation in theano":
            h = self.h(x)
            #if self.do_dropout:
            #    h = h * (srng.uniform(h.shape) < 0.5)
            h_r = h.reshape([h.shape[0], self.nblocks, self.block_size])
            masked = h_r * mask.dimshuffle(0,1,'x')
            masked = masked.reshape(h.shape)

        self.sample_probs = T.prod(mask*probs+(1-probs)*(1-mask), axis=1)
        self.probs = probs
        return masked, mask

    
class ConvLayer:
    def __init__(self, filter_shape, #image_shape,
                 activation = lambda x:x,
                 use_bias=False,
                 init="glorot",
                 inputDropout=None):
        #print "in:",image_shape,"kern:",filter_shape,
        #mw = (image_shape[2] - filter_shape[2]+1)
        #mh = (image_shape[3] - filter_shape[3]+1)
        #self.output_shape = (image_shape[0], filter_shape[0], mw, mh)
        #print "out:",self.output_shape,
        #print "(",numpy.prod(self.output_shape[1:]),")"

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) )
        self.filter_shape = filter_shape
        #self.image_shape = image_shape
        self.activation = activation
        if init=="glorot":
            self.k = k = numpy.sqrt(6./(fan_in+fan_out))
        elif init=="bengio":
            if inputDropout:
                self.k = k = numpy.sqrt(6.*inputDropout/(fan_in))
            else:
                self.k = k = numpy.sqrt(6./(fan_in))

        self.use_bias = use_bias

        W = shared('Wconv', filter_shape, 'uniform', k=k)
        self.W = W
        if use_bias:
            b = shared('bconv',filter_shape[0], 'zero')
            self.b = b

    def __call__(self, x, *args):
        conv_out = conv.conv2d(
            input=x,
            filters=self.W,
            filter_shape=self.filter_shape,
            #image_shape=self.image_shape if image_shape is None else image_shape
        )
        if self.use_bias:
            out = self.activation(conv_out + self.b.dimshuffle('x',0,'x','x'))
        else:
            out = self.activation(conv_out)
        return out

class Maxpool:
    def __init__(self,ds,ignore_border=False):
        self.ds = ds
        self.ib = ignore_border
    def __call__(self, x, *args):
        return max_pool_2d(x, self.ds,ignore_border=self.ib)


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img

class DeconvLayer:
    def __init__(self, filter_shape,
                 activation = T.nnet.sigmoid):
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) )
        self.filter_shape = filter_shape
        self.activation = activation
        self.k = k = numpy.sqrt(6./(fan_in+fan_out))

        W = shared('Wdeconv', filter_shape, 'uniform', k=k)
        #b = shared('bdeconv',filter_shape[0], 'zero')
        self.W = W
        #self.b = b

    def __call__(self, x, *args):
        # that doesn't seem to work...
        #conv_out = deconv(x, self.W, subsample=(2,2), border_mode=(2, 2))

        if 1:
            conv_out = conv.conv2d(
                input=x,
                filters=self.W,
                filter_shape=self.filter_shape,
                #image_shape=self.image_shape if image_shape is None else image_shape,
                border_mode='full'
            )
        return self.activation(conv_out)# + self.b.dimshuffle('x',0,'x','x'))

class Upscale:
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, x):
        w,h = self.scale
        ox = x
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1, x.shape[3], 1))
        o = T.ones((1, 1, 1, w, 1, h))
        x = (x * o).reshape((ox.shape[0], ox.shape[1], ox.shape[2]*w, ox.shape[3]*h))
        return x

class StackModel:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, *x, **kw):
        activations = kw.get('activations', None)
        upto = kw.get('upto', len(self.layers))

        for i,l in enumerate(self.layers[:upto]):
            x = l(*x)
            if activations is not None: activations.append(x)
            if type(x) != list and type(x) != tuple:
                x = [x]
            if hasattr(l, "params") and not hasattr(l, "_nametagged"):
                for p in l.params:
                    p.name = p.name+"-"+str(i)
                l._nametagged=True
        return x if len(x)>1 else x[0]

    def reconstruct(self, x, upto):

        h = self(x, upto=upto)
        for l in self.layers[upto-1::-1]:
            h = l.reconstruct(h)
        return h


    def LSUV(self, minibatch, givens={},upto=-1):
        x,y = minibatch
        for i in self.layers:
            if hasattr(i, 'orthonormalize'):
                print "Orthonormalize",i
                i.orthonormalize()
            else:
                print "skip",i
        for i,l in enumerate(self.layers[:upto]):
            print i,l
            if not hasattr(l,'W'):
                print "skip"
                continue
            X = T.matrix()
            output = self(X,upto=i+1)
            updates = {l.W: l.W / T.sqrt(T.var(output))}
            f = theano.function([X],[T.sqrt(T.var(output)), T.var(output)], updates=updates,
                                givens=givens,on_unused_input='ignore')
            for k in range(10):
                svar, var = f(x)
                print var, abs(var-1)
                print abs(l.W.get_value()).mean()
                if abs(var-1) <= 0.01: break

class LSTM:
    def __init__(self,
                 n_in,
                 n_hidden):
        W_i = shared('W_i', (n_in,n_hidden))
        U_i = shared('U_i', (n_hidden,n_hidden))
        b_i = shared('b_i', (n_hidden,),'zero')

        W_c = shared('W_c', (n_in,n_hidden))
        U_c = shared('U_c', (n_hidden,n_hidden))
        b_c = shared('b_c', n_hidden)

        W_f = shared('W_f', (n_in,n_hidden))
        U_f = shared('U_f', (n_hidden,n_hidden))
        b_f = shared('b_f', (n_hidden,),'zero')

        W_o = shared('W_o', (n_in,n_hidden))
        U_o = shared('U_o', (n_hidden,n_hidden))
        V_o = shared('V_o', (n_hidden,n_hidden))
        b_o = shared('b_o', (n_hidden,),'zero')

        def apply(x, htm1, Ctm1):
            i_t = T.nnet.sigmoid(T.dot(x,W_i) + T.dot(htm1, U_i) + b_i)
            C_tilde = T.tanh(T.dot(x,W_c) + T.dot(htm1, U_c)+b_c)
            f_t = T.nnet.sigmoid(T.dot(x,W_f) + T.dot(htm1, U_f) + b_f)
            C_t = i_t * C_tilde + f_t * Ctm1
            o_t = T.nnet.sigmoid(T.dot(x,W_o) +
                                 T.dot(htm1,U_o) +
                                 T.dot(C_t,V_o) + b_o)
            h_t = o_t * T.tanh(C_t)
            return h_t, C_t
        self.apply = apply

# additive memory recurrent unit
class AMRU:
    def __init__(self,
                 n_in,
                 n_hidden,
                 n_memory,
                 activation=T.nnet.sigmoid):
        W_x = shared('W_x', (n_in,n_hidden))
        W_h = shared('W_h', (n_hidden,n_hidden))
        U_m = shared('U_m', (n_memory,n_hidden))
        b_h = shared('b_h', (n_hidden,),'zero')

        W_m = shared('W_m', (n_hidden,n_memory))

        def apply(x, htm1, mtm1):
            ht = activation( T.dot(x, W_x) + T.dot(htm1, W_h) +
                             T.dot(mtm1, U_m) + b_h)
            mt = T.tanh(mtm1 + T.dot(ht, W_m))
            return ht, mt
        self.apply = apply

class GenerativeAMRU:
    def __init__(self,
                 n_hidden,
                 n_memory):
        W_h = shared('W_h', (n_hidden,n_hidden))
        U_m = shared('U_m', (n_memory,n_hidden))
        b_h = shared('b_h', (n_hidden,),'zero')

        W_m = shared('W_m', (n_hidden,n_memory))

        def apply(htm1, mtm1):
            ht = T.nnet.sigmoid( T.dot(htm1, W_h) +
                                 T.dot(mtm1, U_m) + b_h)
            mt = T.tanh(mtm1 + T.dot(ht, W_m))
            return ht, mt
        self.apply = apply


def reinforce_no_baseline(params, policy, cost, lr, regularising_cost = None, log_pol=None):
    """
    return reinforce updates
    @policy and @cost should be of shape (minibatch_size, 1)
    @policy should be the probability of the sampled actions
    """
    if log_pol is None:
        log_pol = T.log(policy)
    if regularising_cost is None:
        return [(i, i - lr * gi) for i,gi in
                zip(params, T.Lop(f=log_pol, wrt=params, eval_points=cost))]
    else:
        return [(i, i - lr * (gi+gr)) for i,gi,gr in
                zip(params,
                    T.Lop(f=log_pol, wrt=params, eval_points=cost),
                    T.grad(regularising_cost, params))]


def reinforce_no_baseline_momentum(params, policy, cost, epsilon, lr, regularising_cost = None):
    """
    return reinforce updates
    @policy and @cost should be of shape (minibatch_size, 1)
    @policy should be the probability of the sampled actions
    """
    log_pol = T.log(policy)
    if regularising_cost is None:
        raise ValueError()
        return [(i, i - lr * gi) for i,gi in
                zip(params, T.Lop(f=log_pol, wrt=params, eval_points=cost))]
    else:
        return momentum(params,
                        [gi+gr
                         for gi,gr in zip(T.Lop(f=log_pol, wrt=params, eval_points=cost),
                                          T.grad(regularising_cost, params))],
                        epsilon,
                        lr)




class GenericClassificationDataset:
    def __init__(self, which, alt_path=None, doDiv255=False):
        self.alt_path = alt_path
        self.doDiv255 = doDiv255
        if which == "mnist":
            self.load_mnist()
        elif which == "cifar10":
            self.load_cifar10()
        elif which == "svhn":
            self.load_svhn()
        elif which == "covertype":
            self.load_covertype()
        else:
            raise ValueError("Don't know about this dataset: '%s'"%which)

        if self.doDiv255:
            self.pp = lambda x,y:(numpy.float32(x/255.), y)
        else:
            self.pp = lambda x,y:(x,y)

    def load_covertype(self):
        f = gzip.open(self.alt_path if self.alt_path else "/data/UCI/covtype.pkl.gz", 'rb')
        numpy.random.seed(142857)
        data = cPickle.load(f)
        X = data[:,:-1]
        Y = data[:,-1] - 1 # 1-based labels -> 0-based
        
        # these numbers are from the dataset instructions,
        # they incur a perfect balance of the training classes in the dataset
        # but then the test set is 50x bigger than the training set...?
        # From looking at the data:
        #   Y = numpy.int32(Y)
        #   print numpy.bincount(Y[:11340])
        #   print numpy.bincount(Y[11340:11340+3780])
        #   print numpy.bincount(Y[11340+3780:])
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
        f = gzip.open(self.alt_path if self.alt_path else "/data/mnist.pkl.gz", 'rb')
        self.train,self.valid,self.test = map(list,cPickle.load(f))
        f.close()

        #self.train[0] = self.train[0]/self.train[0].std(axis=0) - self.train[0].mean()
        #self.valid[0] = self.valid[0]/self.valid[0].std(axis=0) - self.valid[0].mean()
        #self.test[0] = self.test[0]/self.test[0].std(axis=0) - self.test[0].mean()
        self.train[1] = numpy.uint8(self.train[1])
        self.valid[1] = numpy.uint8(self.valid[1])
        self.test[1] = numpy.uint8(self.test[1])

    def load_cifar10(self):
        trainX, trainY, testX, testY = cPickle.load(file(self.alt_path if self.alt_path else '/data/cifar/cifar_10_shuffled.pkl','r'))
        trainX = numpy.float32(trainX / 255.)
        testX = numpy.float32(testX / 255.)
        print testX.shape, trainX.shape
        print testX.mean(),trainX.mean()
        self.train = [trainX[:40000], trainY[:40000]]
        self.valid = [trainX[40000:], trainY[40000:]]
        self.test = [testX, testY]

    def load_svhn(self):
        train, test = cPickle.load(file(self.alt_path,'r'))
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
    def export_simple_plot1d(ys,path,ylabel="",format=None):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        pyplot.clf()
        pyplot.plot(numpy.arange(len(ys)), ys)
        pyplot.show(block=False)
        if ylabel: pyplot.ylabel(ylabel)
        pyplot.savefig(path, format=format)

    @staticmethod
    def export_multi_plot1d(ys,path,ylabel="",format=None):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        pyplot.clf()
        for i in ys:
            pyplot.plot(numpy.arange(len(i)), i)
        pyplot.show(block=False)
        if ylabel: pyplot.ylabel(ylabel)
        pyplot.savefig(path, format=format)

    @staticmethod
    def open_video(path, method='avconv', outputsize='800x600', fps=60):
        from subprocess import Popen, PIPE
        video = Popen([method, '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg',
                       '-r', str(fps),'-s',outputsize, '-i', '-',
                       '-qscale', '9', '-r', str(fps), path], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        return video
