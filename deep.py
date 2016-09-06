from __future__ import print_function
import numpy
import gzip
import pickle# as pickle
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .sparse_dot import*
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool

def momentum(epsilon):
    def gradient_descent(params, grads, lr):
        mom_ws = [theano.shared(0*(i.get_value()+1), i.name+" momentum")
                  for i in params]
        mom_up = [(i, epsilon * i + (1-epsilon) * gi)
                  for i,gi in zip(mom_ws, grads)]
        up = [(i, i - lr * mi) for i,mi in zip(params, mom_ws)]
        return up+mom_up
    return gradient_descent

def delta_descent(delta):
    def gradient_descent(params, grads, lr):
        up = [(i, i - lr * T.sgn(gi)*delta) for i,gi in zip(params, grads)]
        return up
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

def rmsprop(decay, epsilon=1e-3, clip=None):
    def sgd(params, grads, lr):
        if clip is not None:
            grads = [T.clip(i, -clip, clip) for i in grads]
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

            new_last_1_moment = T.cast((numpy.float32(1.) - self.b1) * grad + self.b1 * last_1_moment, 'float32')
            new_last_2_moment = T.cast((numpy.float32(1.) - self.b2) * grad**2 + self.b2 * last_2_moment, 'float32')

            updates[last_1_moment] = new_last_1_moment
            updates[last_2_moment] = new_last_2_moment
            updates[param] = (param - (lr * (new_last_1_moment / (numpy.float32(1.) - self.b1**t)) /
                                      (T.sqrt(new_last_2_moment / (numpy.float32(1.) - self.b2**t)) + self.eps)))

        return list(updates.items())


class SharedGenerator:
    def __init__(self):
        self.reset()
        self.init_tensor_x = T.scalar()
        self.init_minibatch_x = numpy.float32(0)
        self.isJustReloadingModel = False
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
    def bindNew(self, name='default'):
        p = []
        self.bind(p, name)
        return p

    def __call__(self, name, shape, init='uniform', **kwargs):
        #print("init",name,shape,init,kwargs)
        if type(init).__module__ == numpy.__name__: # wtf numpy
            values = init
        elif init == "uniform" or init == "glorot":
            k = numpy.sqrt(6./numpy.sum(shape)) if 'k' not in kwargs else kwargs['k']
            values = numpy.random.uniform(-k,k,shape)
        elif init == "bengio" or init == 'relu':
            p = kwargs['inputDropout'] if 'inputDropout' in kwargs and kwargs['inputDropout'] else 1
            k = numpy.sqrt(6.*p/shape[0]) if 'k' not in kwargs else kwargs['k']
            values = numpy.random.uniform(-k,k,shape)
        elif init == "one":
            values = numpy.ones(shape)
        elif init == "zero":
            values = numpy.zeros(shape)
        elif init == 'ortho':
            def sym(w):
                import numpy.linalg
                from scipy.linalg import sqrtm, inv
                return w.dot(inv(sqrtm(w.T.dot(w))))
            values = numpy.random.normal(0,1,shape)
            values = sym(values).real

        else:
            print(type(init))
            raise ValueError(init)
        s = theano.shared(numpy.float32(values), name=name)
        self.param_list.append(s)
        return s

    def exportToFile(self, path):
        exp = {}
        for g in self.param_groups:
            exp[g] = [i.get_value() for i in self.param_groups[g]]
        pickle.dump(exp, open(path,'wb'), -1)

    def importFromFile(self, path):
        exp = pickle.load(open(path,'rb'))
        for g in exp:
            for i in range(len(exp[g])):
                print(g, exp[g][i].shape)
                self.param_groups[g][i].set_value(exp[g][i])
    def attach_cost(self, name, cost):
        self.param_costs[name].append(cost)
    def get_costs(self, name):
        return self.param_costs[name]
    def get_all_costs(self):
        return [j  for i in self.param_costs for j in self.param_costs[i]]
    def get_all_names(self):
        print([i for i in self.param_costs])
        print(self.param_costs.keys())
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

from theano.ifelse import ifelse

isTestTime = theano.shared(numpy.float32(0), 'isTestTime')
one = T.as_tensor_variable(numpy.float32(1))
zero = T.as_tensor_variable(numpy.float32(0))
if 'gpu' in theano.config.device:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as SRNG
else:
    from theano.tensor.shared_randomstreams import RandomStreams as SRNG
    # ConvLayer.use_cudnn = False

srng = SRNG(12345)

relu = lambda x: T.maximum(0,x)
def lrelu():
    theta = shared('lrelu leak', (1,), numpy.float32(0.01))
    return lambda x: T.maximum(theta*x,x)

def dropout(p):
    return lambda x: ifelse(isTestTime,
                            x,
                            x * (srng.uniform(x.shape) < p) / p)
linear = lambda x:x
Id = lambda x:x




class ConvBatchNormalization:
    def __init__(self, n):
        self.n = n
        self.gamma = shared('gamma', (n,), "one")
        self.beta = shared('beta', (n,), "zero")
    def __call__(self, x, *args):
        means = T.mean(x, axis=[0,2,3]).dimshuffle('x',0,'x','x')
        std = (T.maximum(T.std(x, axis=[0,2,3]),numpy.float32(1e-6))).dimshuffle('x',0,'x','x')
        out = (x - means) / std
        out = self.gamma.dimshuffle('x',0,'x','x') * out + self.beta.dimshuffle('x',0,'x','x')
        return out

class HiddenLayer:
    def __init__(self, n_in, n_out, activation, init="glorot", canReconstruct=False,
                 inputDropout=None,name="",outputBatchNorm=False):
        self.W = shared("W"+name, (n_in, n_out), init, inputDropout=inputDropout)
        self.b = shared("b"+name, (n_out,), "zero")
        self.activation = activation
        self.params = [self.W, self.b]
        if canReconstruct:
            self.bprime = shared("b'"+name, (n_in,), "zero")
            self.params+= [self.bprime]
        if outputBatchNorm:
            self.gamma = shared('gamma', (n_out,), "one")
            self.beta = shared('beta', (n_out,), "zero")
            self.params += [self.gamma, self.beta]
            def bn(x):
                mu = x.mean(axis=0)
                std = T.maximum(T.std(x, axis=0),numpy.float32(1e-6))
                x = (x - mu) / std
                return activation(self.gamma * x + self.beta)
            self.activation = bn
        self.name=name
    def orthonormalize(self):
        import numpy.linalg
        from scipy.linalg import sqrtm, inv

        def sym(w):
            return w.dot(inv(sqrtm(w.T.dot(w))))
        Wval = numpy.random.normal(0,1,self.W.get_value().shape)
        Wval = sym(Wval).real
        self.W.set_value(numpy.float32(Wval))

    def apply(self, *x):
        return self(*x)
    def __call__(self, x, *args):
        return self.activation(T.dot(x,self.W) + self.b)
    def apply_partially(self, x, nin, nout):
        return self.activation(T.dot(x, self.W[:nin,:nout]) + self.b[:nout])
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
        print(xmask)
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
    use_cudnn = True
    def __init__(self, filter_shape, #image_shape,
                 activation = lambda x:x,
                 use_bias=True,
                 init="glorot",
                 inputDropout=None,
                 normalize=False,
                 mode="valid",
                 stride=(1,1),
                 use_cudnn=None):
        #print("in:",image_shape,"kern:",filter_shape,)
        #mw = (image_shape[2] - filter_shape[2]+1)
        #mh = (image_shape[3] - filter_shape[3]+1)
        #self.output_shape = (image_shape[0], filter_shape[0], mw, mh)
        #print("out:",self.output_shape,)
        #print("(",numpy.prod(self.output_shape[1:]),")")
        self.normalize = normalize
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) )
        self.filter_shape = filter_shape
        #self.image_shape = image_shape
        self.activation = activation
        self.mode=mode
        if use_cudnn is not None:
            self.cudnn= use_cudnn
        else:
            self.cudnn = ConvLayer.use_cudnn
        self.stride = stride

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
        self.params = [W]
        if use_bias:
            b = shared('bconv',filter_shape[0], 'zero')
            self.b = b
            self.params += [b]

        if self.normalize:
            g = shared('normalization_g', (filter_shape[0],), 'one')
            self.g = g
            self.params += [g]

    def __call__(self, x, *args):
        if self.normalize:
            W = self.g.dimshuffle(0,'x','x','x') * \
                (self.W - self.W.mean(axis=[1,2,3]).dimshuffle(0,'x','x','x')) /  \
                T.sqrt(T.sum(self.W**2, axis=[1,2,3])).dimshuffle(0,'x','x','x')
        else:
            W = self.W
        #print("conv call:",x,W,self.mode,self.stride)
        #print(x.tag.test_value.shape
        #try:
        #    print(W.tag.test_value.shape)
        #except:
        #    print(W.get_value().shape)
        #print(self.mode)
        #print(self.stride)
        if self.cudnn:
            conv_out = dnn_conv(x,W,self.mode,self.stride)
        else:
            if self.mode == 'half' and 'cpu' in theano.config.device:
                fso = self.filter_shape[2] - 1
                nps = x.shape[2]
                conv_out = conv.conv2d(input=x, filters=W,
                                       filter_shape=self.filter_shape,
                                       border_mode='full',
                                       subsample=self.stride)[:,:,fso:nps+fso,fso:nps+fso]
            else:
                conv_out = conv.conv2d(
                    input=x,
                    filters=W,
                    filter_shape=self.filter_shape,
                    border_mode=self.mode,
                    subsample=self.stride,
                    #image_shape=self.image_shape if image_shape is None else image_shape
                )

        if self.normalize and not shared.isJustReloadingModel:
            mu = T.mean(conv_out, axis=[0,2,3]).eval({shared.init_tensor_x: shared.init_minibatch_x})
            sigma = T.std(conv_out, axis=[0,2,3]).eval({shared.init_tensor_x: shared.init_minibatch_x})
            print("normalizing:",mu.mean(),sigma.mean())
            self.g.set_value( 1 / sigma)
            self.b.set_value(-mu/sigma)

        if hasattr(shared, 'preactivations'):
            shared.preactivations.append(conv_out)

        if 0: # mean-norm
            conv_out = conv_out - conv_out.mean(axis=[0,2,3]).dimshuffle('x',0,'x','x')

        if self.use_bias:
            out = self.activation(conv_out + self.b.dimshuffle('x',0,'x','x'))
        else:
            out = self.activation(conv_out)
        #print("out:", out.tag.test_value.shape)


        return out

class Maxpool:
    def __init__(self,ds,ignore_border=False):
        self.ds = ds
        self.ib = ignore_border
    def __call__(self, x, *args):
        return pool_2d(x, self.ds,ignore_border=self.ib)


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """
    Taken from DCGAN repo:https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py
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


class DoublingDeconvLayer:
    def __init__(self, filter_shape,
                 activation = T.nnet.relu):
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
        if hasattr(x.tag, 'test_value'):
            print("deconv")
            print("x:",x.tag.test_value.shape)
            print("w:",self.W.get_value().shape)
        conv_out = deconv(x, self.W, (2,2), (2, 2))

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
        self.params = set()

    def __call__(self, *x, **kw):
        activations = kw.get('activations', None)
        upto = kw.get('upto', len(self.layers))

        if hasattr(x[0].tag, 'test_value'):
            print("input shape:", x[0].tag.test_value.shape)
        for i,l in enumerate(self.layers[:upto]):
            x = l(*x)
            if hasattr(x,'tag') and hasattr(x.tag, 'test_value'):
                print(l,"output shape:", x.tag.test_value.shape, x.tag.test_value.dtype)
            if activations is not None: activations.append(x)
            if type(x) != list and type(x) != tuple:
                x = [x]
            if hasattr(l, "params") and not hasattr(l, "_nametagged"):
                for p in l.params:
                    p.name = p.name+"-"+str(i)
                    self.params.add(p)
                l._nametagged=True
        self.params = list(self.params)
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
                print("Orthonormalize",i)
                i.orthonormalize()
            else:
                print("skip",i)
        for i,l in enumerate(self.layers[:upto]):
            print(i,l)
            if not hasattr(l,'W'):
                print("skip")
                continue
            X = T.matrix()
            output = self(X,upto=i+1)
            updates = {l.W: l.W / T.sqrt(T.var(output))}
            f = theano.function([X],[T.sqrt(T.var(output)), T.var(output)], updates=updates,
                                givens=givens,on_unused_input='ignore')
            for k in range(10):
                svar, var = f(x)
                print(var, abs(var-1))
                print(abs(l.W.get_value()).mean())
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
