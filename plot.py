from matplotlib import pyplot as pp
import numpy


def smooth_fill(pts, N = 40, label=None,
                edgecolor='#1b2acc', facecolor='#1b10aa',plotPts=True):
    # N is the number of bins
    # n is the number of points per bin
    n = len(pts) / N
    npts = numpy.float32(pts[:len(pts) - (len(pts)%n)]).reshape((-1,n)).T
    means = numpy.mean(npts, axis=0)
    std = numpy.std(npts, axis=0)
    Npts = len(pts)
    if len(pts) < 1000000 and plotPts:
        pp.plot(pts,alpha=0.15)

    below = means-std
    above = means+std
    pp.plot(numpy.linspace(0,Npts,len(means)),means, label=label)
    pp.fill_between(numpy.linspace(0,Npts,len(means)), below, above,
                    alpha=0.2,edgecolor=edgecolor,facecolor=facecolor,antialiased=True)

    #if len(pts) < 1000000:
    #    pp.gca().set_ylim([numpy.min(pts[-len(pts)/2:]), numpy.max(pts[-len(pts)/2:])])
    #else:
    #    pp.gca().set_ylim([numpy.min(below[-len(below)/2:]), numpy.max(above[-len(above)/2:])])

def smooth(pts, plotPts=False, N = 200, label=None, setLims=False, **kwargs):
    # N is the number of bins
    # n is the number of points per bin
    n = len(pts) / N
    if n > 0:
        npts = numpy.float32(pts[:len(pts) - (len(pts)%n)]).reshape((-1,n)).T
        means = numpy.mean(npts, axis=0)
    else:
        means = pts
    Npts = len(pts)
    if len(pts) < 1000000 and plotPts:
        pp.plot(pts,alpha=0.15)

    pp.plot(numpy.linspace(0,Npts,len(means)),means, label=label, **kwargs)

    if len(pts) < 1000000 and setLims:
        pp.gca().set_ylim([numpy.min(pts[-len(pts)/2:]), numpy.max(pts[-len(pts)/2:])])
