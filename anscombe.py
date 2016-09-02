import imageio
from matplotlib import pyplot as plt
from matplotlib import style
import theano
import seaborn as sns
import theano.tensor as T
import numpy as np
from pdselect import pdselect
style.use('seaborn-white')

anscombe = sns.load_dataset('anscombe')
def get_data(dataset):
    sel = pdselect(anscombe, dataset=dataset)
    return np.concatenate([sel.x.values, sel.y.values])
d1 = get_data('I')
d2 = get_data('II')
d3 = get_data('III')
d4 = get_data('IV')

# We switch the order so that the top-right point in d4 matches the right most
# point in the other datasets:
d4[7],d4[5] = d4[5],d4[7]
d4[7+11],d4[5+11] = d4[5+11],d4[7+11]


N_POINTS = len(d1)//2
gamma1 = 10.
X_norm = 9.
X_var = d1[:N_POINTS].var()
Y_norm = 7.50
Y_var = d1[N_POINTS:].var()
nabla = .001

gamma1 = 1.
gamma2 = 1.
gamma3 = .5

def build_error_grad(target):
    current = T.vector('current')
    x = current[:N_POINTS]
    y = current[N_POINTS:]
    error = T.mean( (current - target)**2. ) + \
            T.mean( T.abs_(current - target) ) + \
                gamma1 * (T.mean(y) - Y_norm )**2 + \
                gamma1 * (T.mean(x) - X_norm )**2 + \
                gamma2 * (T.mean((y - Y_norm)**2) - Y_var)**2 + \
                gamma2 * (T.mean((x - X_norm)**2) - X_var)**2 + \
                gamma3 * (np.dot(target[:N_POINTS] - X_norm, target[N_POINTS:] - Y_norm) - T.dot(x - X_norm, y - Y_norm))**2.
    return theano.function([current], error), theano.function([current], T.grad(error, current))


def fig2np(fig):
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape((h,w,3))

fig,ax = plt.subplots()
sns.despine(fig)
currentval = d1.copy()
with imageio.get_writer('anscombe.gif', mode='I') as writer:
    for target in [d2, d3, d4, d1]:
        e,g = build_error_grad(target)
        while np.max(np.abs(currentval-target)) > .1:
            ax.clear()
            x = currentval[:N_POINTS]
            y = currentval[N_POINTS:]
            (x1,x0),_,_,_ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y)
            ax.plot([4,20], [x0+4*x1, x0+20*x1], '-', c='#000099')

            ax.plot([4,20], [y.mean(), y.mean()], 'k-')
            ax.plot([4,20], [y.mean() + y.std(), y.mean() + y.std()], 'k:')
            ax.plot([4,20], [y.mean() - y.std(), y.mean() - y.std()], 'k:')
            ax.scatter(x, y, c='#990000', s=40)

            ax.set_ylim((2.5,13.))
            ax.set_xlim((3.,21.))
            writer.append_data(fig2np(fig))
            for i in range(1000):
                currentval -= nabla * g(currentval)

