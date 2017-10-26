# import numpy as np
# import pylab
#
#
# def wiener():
#     T = 1
#     N = 500
#     t, dt = np.linspace(0, T, N+1, retstep=True)
#     print dt
#     dW = np.random.normal(0., np.sqrt(dt), N+1)
#     dW[0] = 0.
#     W = np.cumsum(dW)
#
#
#
#
#
# theta = 0.15
# N = 5
# mu = 0.
# sigma = 0.1
# dt = 5e-2
# t = np.linspace(0., 10, 10./dt)
# x0 = 0.
# W = np.zeros((N, len(t)))
#
# for i in range(len(t)-1):
#     W[:, i+1] = W[:, i]+np.sqrt(np.exp(2*theta*t[i+1])-np.exp(2*theta*t[i]))*np.random.normal(size=(N,))
#
#
# ex = np.exp(-theta*t);
# x = x0 * ex + mu * (1-ex) + sigma*ex*W/np.sqrt(2*theta)
#
#
# pylab.plot(x[0], '-')
# pylab.show()
# pylab.plot(x[1], '-')
# pylab.show()
#
# def step(t, theta, sigma, mu, history):
#     pass

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')