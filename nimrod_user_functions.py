#!/usr/local/sci/bin/python2.7
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


class FileLoader(object):
    def __init__(self, filelist, chilbolton_centred=False):
        self.filelist = filelist
        self.chilbolton_centred = chilbolton_centred
        self.curr_file = 0
        self.curr_index = 0
        self.curr_da = xr.open_dataarray(self.filelist[0])
        # Chilbolton coords in OSGB eastings/northings.
        self.chil_idx = (np.abs(self.curr_da.eastings.data - 439285)).argmin()
        self.chil_idy = (np.abs(self.curr_da.northings.data - 138620)).argmin()

    def load_next(self):
        while True:
            self.curr_index += 1
            if self.curr_index >= self.curr_da.shape[0]:
                self.curr_file += 1
                if self.curr_file >= len(self.filelist):
                    break
                self.curr_da = xr.open_dataarray(self.filelist[self.curr_file])
                self.curr_index = 0
            time = pd.to_datetime(self.curr_da.time[self.curr_index].item())
            fidd = '{}_{}'.format(os.path.basename(self.filelist[self.curr_file]), f'{time.hour:02}:{time.minute:02}')
            if self.chilbolton_centred:
                yield self.curr_da[self.curr_index,
                                   self.chil_idy - 300:self.chil_idy + 300,
                                   self.chil_idx - 400:self.chil_idx + 400].data, fidd, time
            else:
                yield self.curr_da[self.curr_index, 750:1350, 600:1400].data, fidd, time


###################################################
# timediff IS A USER SPECIFIED FUNCION TO CALCULATE TIME SEPARATION BETWEEN CONSECUTIVE IMAGES
# OUTPUT
# tdif = time difference in units relevant to user specification (to be divided by "dt" in wrapper.py)
###################################################

def timediff(oldh, oldm, newh, newm):
    hdif = newh - oldh
    mdif = newm - oldm
    tdif = 60. * hdif + mdif

    return tdif


###################################################
# plot_example IS ONLY USED AS AN ILLUSTRATION
# OF THE EXAMPLE DATA
###################################################

def plot_example(write_file_ID, nt, rain, xmat, ymat, newumat, newvmat, num_dt, wasarray, lifearray, threshold,
                 IMAGES_DIR, do_vectors):
    '''
    PLOT FIGURES WITH RAINFALL RATE AND STORM LABELS
    FOR ILLUSTRATIVE AND TESTING PURPOSES
    '''

    lrain = rain + 0.0
    lrain[np.where(lrain <= 0.)] = 0.01

    figa = plt.figure(figsize=(6, 7))
    # ax = figa.add_subplot(111)
    # con = ax.imshow(f, cmap=cm.jet, interpolation='nearest')
    con = plt.pcolor(xmat, ymat, np.log2(lrain), vmin=-1, vmax=5)
    plt_ax = plt.gca()
    left, bottom, width, height = plt_ax.get_position().bounds
    posnew = [left, bottom + height / 7, width, width * 6 / 7]
    plt_ax.set_position(posnew)
    plt.xlabel('Distance from Chilbolton [km]')
    plt.ylabel('Distance from Chilbolton [km]')
    colorbar_axes = figa.add_axes([left, bottom, width, 0.01])
    # add a colourbar with a label
    cbar = plt.colorbar(con, colorbar_axes, orientation='horizontal')
    cbar.set_label('Rainfall rate [log2 mm hr^{-1}]')
    plt.savefig(IMAGES_DIR + 'Rainrate_' + write_file_ID + '.png')
    plt.close()

    figb = plt.figure(figsize=(6, 7))
    # ax = figa.add_subplot(111)
    # con = ax.imshow(f, cmap=cm.jet, interpolation='nearest')
    con = plt.pcolor(xmat, ymat, wasarray, vmin=0)
    plt_ax = plt.gca()
    left, bottom, width, height = plt_ax.get_position().bounds
    posnew = [left, bottom + height / 7, width, width * 6 / 7]
    plt_ax.set_position(posnew)
    plt.xlabel('Distance from Chilbolton [km]')
    plt.ylabel('Distance from Chilbolton [km]')
    colorbar_axes = figb.add_axes([left, bottom, width, 0.01])
    # add a colourbar with a label
    cbar = plt.colorbar(con, colorbar_axes, orientation='horizontal')
    cbar.set_label('Storm ID')
    plt.savefig(IMAGES_DIR + 'Stormid_' + write_file_ID + '.png')
    plt.close()

    lifearray[np.where(lifearray == 0)] = -6
    figc = plt.figure(figsize=(6, 7))
    # ax = figa.add_subplot(111)
    # con = ax.imshow(f, cmap=cm.jet, interpolation='nearest')
    con = plt.pcolor(xmat, ymat, 5 * lifearray, vmin=-30, vmax=60)
    plt_ax = plt.gca()
    left, bottom, width, height = plt_ax.get_position().bounds
    posnew = [left, bottom + height / 7, width, width * 6 / 7]
    plt_ax.set_position(posnew)
    plt.xlabel('Distance from Chilbolton [km]')
    plt.ylabel('Distance from Chilbolton [km]')
    colorbar_axes = figc.add_axes([left, bottom, width, 0.01])
    # add a colourbar with a label
    cbar = plt.colorbar(con, colorbar_axes, orientation='horizontal')
    cbar.set_label('Life time [mins]')
    plt.savefig(IMAGES_DIR + 'Lifetime_' + write_file_ID + '.png')
    plt.close()
    if do_vectors == True:
        figd = plt.figure(figsize=(6, 7))
        # ax = figa.add_subplot(111)
        # con = ax.imshow(f, cmap=cm.jet, interpolation='nearest')
        con = plt.contour(xmat, ymat, lrain, levels=[threshold])
        plt.quiver(xmat[::10, ::10], ymat[::10, ::10], newumat[::10, ::10] / num_dt, newvmat[::10, ::10] / num_dt,
                   pivot='mid', units='width')
        plt_ax = plt.gca()
        left, bottom, width, height = plt_ax.get_position().bounds
        posnew = [left, bottom + height / 7, width, width * 6 / 7]
        plt_ax.set_position(posnew)
        plt.xlabel('Distance from Chilbolton [km]')
        plt.ylabel('Distance from Chilbolton [km]')
        plt.savefig(IMAGES_DIR + 'Vectors_' + write_file_ID + '.png')
        plt.close()
