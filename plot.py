import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_brain(data, mask):
    temp = np.zeros(mask.shape)
    temp[mask] = data
    plt.imshow(temp[:, :, 40])
    plt.colorbar()

def stack(temp, x,y,z):

    sx = np.flipud(np.squeeze(temp[x, :, :]).T)
    sy = np.flipud(np.squeeze(temp[:, y, :]).T)
    sz = np.flipud(np.squeeze(temp[:, :, z]).T)

    upper = np.hstack((sy,sx))
    lower = np.hstack((sz, np.zeros((sz.shape[0], sx.shape[1]))))
    both = np.vstack((upper,lower))

    return both

def plot_source(source, template, mask, th=3, vmin=-5, vmax=5):
    '''
    INPUT:
    source - 1d vector with weight values
    template - 3d array with background image
    mask - 1d vector with indices that map the source to 3d
    th - threshold to plot on top of background
    '''
    
    # find maximum position
    shape = template.shape

    def unravel(x): return np.unravel_index(x, shape)

    temp = np.zeros(shape)
    temp[mask] = source
    x, y, z = unravel(np.abs(temp).argmax())
    
    source = stack(temp, x,y,z)
    
    sm = np.ma.masked_where(np.abs(source) < th, source)

    background = stack(template, x,y,z)

    plt.imshow(background, cmap=cm.gray)
    plt.imshow(sm, cmap=cm.jet, vmin=vmin, vmax=vmax, aspect='equal')
    plt.axis('off')
