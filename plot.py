import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_brain(data, mask):
    temp = np.zeros(mask.shape)
    temp[mask] = data
    plt.imshow(temp[:, :, 40])
    plt.colorbar()


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
    temp[unravel(mask)] = source
    x, y, z = unravel(np.abs(temp).argmax())
    sx = np.squeeze(temp[x, :, :])
    sy = np.squeeze(temp[:, y, :])
    sz = np.squeeze(temp[:, :, z])

    # draw template on maximum position
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(template[:, :, z]), cmap=cm.gray)
    masked_data = np.ma.masked_where(np.abs(sz) < th, sz)
    plt.imshow(masked_data, cmap=cm.jet, interpolation='none',
               vmin=vmin, vmax=vmax)

    plt.subplot(1, 3, 1)
    plt.imshow(np.flipud(np.squeeze(template[:, y, :]).T), cmap=cm.gray)
    sy = np.flipud(sy.T)
    masked_data = np.ma.masked_where(np.abs(sy) < th, sy)
    plt.imshow(masked_data, cmap=cm.jet, interpolation='none',
               vmin=vmin, vmax=vmax)

    plt.subplot(1, 3, 2)
    plt.imshow(np.flipud(np.squeeze(template[x, :, :]).T), cmap=cm.gray)
    sx = np.flipud(sx.T)
    masked_data = np.ma.masked_where(np.abs(sx) < th, sx)
    plt.imshow(masked_data, cmap=cm.jet,
               interpolation='none', vmin=vmin,
               vmax=vmax)
