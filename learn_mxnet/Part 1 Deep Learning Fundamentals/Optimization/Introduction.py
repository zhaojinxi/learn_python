import numpy
import matplotlib.pyplot

def f(x):
    return x * numpy.cos(numpy.pi * x)

x = numpy.arange(-1.0, 2.0, 0.1)
fig = matplotlib.pyplot.figure()
subplt = fig.add_subplot(111)
subplt.annotate('local minimum', xy=(-0.3, -0.2), xytext=(-0.8, -1.0), arrowprops=dict(facecolor='black', shrink=0.05))
subplt.annotate('global minimum', xy=(1.1, -0.9), xytext=(0.7, 0.1), arrowprops=dict(facecolor='black', shrink=0.05))
matplotlib.pyplot.plot(x, f(x))
matplotlib.pyplot.show()

x = numpy.arange(-2.0, 2.0, 0.1)
fig = matplotlib.pyplot.figure()
subplt = fig.add_subplot(111)
subplt.annotate('saddle point', xy=(0, -0.2), xytext=(-0.4, -5.0), arrowprops=dict(facecolor='black', shrink=0.05))
matplotlib.pyplot.plot(x, x**3)
matplotlib.pyplot.show()