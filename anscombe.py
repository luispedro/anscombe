import imageio
from matplotlib import pyplot as plt
from matplotlib import style
import theano
import seaborn as sns
import theano.tensor as T
import numpy as np
from pdutils import pdselect
style.use('seaborn-white')

anscombe = sns.load_dataset('anscombe')
anscombe1 = pdselect(anscombe, dataset='I')
anscombe2 = pdselect(anscombe, dataset='II')
anscombe3 = pdselect(anscombe, dataset='III')
x = anscombe2.x.values.copy()
y1 = anscombe1.y.values.copy()
y2 = anscombe2.y.values.copy()
y3 = anscombe3.y.values.copy()

gamma1 = 10.
Y_norm = 7.50
Y_var = y1.var()
nabla = .001

gamma1 = 1.
gamma2 = 1.
gamma3 = .5

def build_grad(target):
    current = T.vector('current')
    error = T.mean( (current - target)**2 ) + \
                gamma1 * (T.mean(current) - Y_norm )**2 + \
                gamma2 * (T.mean((current - Y_norm)**2) - Y_var)**2 + \
                gamma3 * (np.dot(x, target - Y_norm) - T.dot(x, current - Y_norm))**2.
    return theano.function([current], T.grad(error,current))

def step(currentval, g):
    currentval = currentval.copy()
    for i in range(1000):
        currentval -= nabla * g(currentval)
    return currentval

def fig2np(fig):
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape((h,w,3))

fig,ax = plt.subplots()
sns.despine(fig, offset=-4)
currentval = y1.copy()
with imageio.get_writer('anscombe.gif', mode='I') as writer:
    for target in [y2, y3, y1]:
        g = build_grad(target)
        while np.max(np.abs(currentval-target)) > .1:
            ax.clear()
            (x1,x0),_,_,_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, currentval)
            ax.plot([4,14], [x0+4*x1, x0+14*x1], '-', c='#000099')

            ax.plot([4,14], [currentval.mean(), currentval.mean()], 'k-')
            ax.plot([4,14], [currentval.mean() + currentval.std(), currentval.mean() + currentval.std()], 'k:')
            ax.plot([4,14], [currentval.mean() - currentval.std(), currentval.mean() - currentval.std()], 'k:')
            ax.scatter(x, currentval, c='#990000', s=40)

            ax.set_ylim((3.,13.))
            writer.append_data(fig2np(fig))
            currentval = step(currentval, g)

