import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_brain(data, mask):
    temp = np.zeros(mask.shape)
    temp[mask] = data
    plt.imshow(temp[:, :, 40])
    plt.colorbar()


def plot_source(source, template, mask, th=3, vmin=-5, vmax=5):

    # find maximum position
    temp = np.zeros(mask.shape)
    temp[mask] = source
    x, y, z = np.where(np.abs(temp) == np.abs(temp).max())
    sx = np.squeeze(temp[x, :, :])
    sy = np.squeeze(temp[:, y, :])
    sz = np.squeeze(temp[:, :, z])

    # draw template on maximum position
    temp = np.zeros(mask.shape)
    temp[mask] = template
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(temp[:, :, z]), cmap=cm.gray)
    masked_data = np.ma.masked_where(np.abs(sz) < th, sz)
    plt.imshow(masked_data, cmap=cm.jet, interpolation='none',
               vmin=vmin, vmax=vmax)

    plt.subplot(1, 3, 1)
    plt.imshow(np.flipud(np.squeeze(temp[:, y, :]).T), cmap=cm.gray)
    sy = np.flipud(sy.T)
    masked_data = np.ma.masked_where(np.abs(sy) < th, sy)
    plt.imshow(masked_data, cmap=cm.jet, interpolation='none',
               vmin=vmin, vmax=vmax)

    plt.subplot(1, 3, 2)
    plt.imshow(np.flipud(np.squeeze(temp[x, :, :]).T), cmap=cm.gray)
    sx = np.flipud(sx.T)
    masked_data = np.ma.masked_where(np.abs(sx) < th, sx)
    plt.imshow(masked_data, cmap=cm.jet,
               interpolation='none', vmin=vmin,
               vmax=vmax)
