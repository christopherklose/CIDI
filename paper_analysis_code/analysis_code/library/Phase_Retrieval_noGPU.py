"""
Python Dictionary for Phase retrieval in Python using functions defined in fth_reconstroction

2020
@authors:   RB: Riccardo Battistelli (riccardo.battistelli@helmholtz-berlin.de)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import numpy.fft as fft
from scipy.fftpack import fft2, ifftshift, fftshift,ifft2
import scipy.io
from scipy.stats import linregress
import fthcore as fth
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets
from skimage.draw import disk
import h5py
import math


#############################################################
#    function for saving Hdf5 file
#############################################################

"""
functions to create and read hdf5 files.
groups will be converted to dictionaries, containing the data
supports nested dictionaries.

to create hdf file:
    create_hdf5(dict0,filename) where dict0 is the dictionary containing the data and filename the file name
to read hdf file:
    data=cread_hdf5(filename) data will be a dictionary containing all information in "filename.hdf5"
riccardo 2020

"""

def read_hdf5(filename, extension=".hdf5", print_option=True):
    
    f = h5py.File(filename+extension, 'r')
    dict_output = readHDF5(f, print_option = print_option, extension=extension)
    
    return dict_output
    
def readHDF5(f, print_option=True, extension=".hdf5", dict_output={}):
    
    for i in f.keys():
        
    
        if type(f[i]) == h5py._hl.group.Group:
            if print_option==True:
                print("### ",i)
                print("---")
            dict_output[i]=readHDF5(f[i],print_option=print_option,dict_output={})
            if print_option==True:
                print("---")
        
        else:
            dict_output[i]=f[i][()]
            if print_option==True:
                print("•",i, "                  ", type(dict_output[i]))
        
        
    return dict_output
    
def create_hdf5(dict0,filename, extension=".hdf5"):
    
    f=createHDF5(dict0,filename, extension=extension)
    f.close()


def createHDF5(dict0,filename, extension=".hdf5",f=None):
    '''creates HDF5 data structures strating from a dictionary. supports nested dictionaries'''
    print(dict0.keys())
    
#    try:
#        f = h5py.File(filename+ ".hdf5", "w")
#        print("ok")
#    except OSError:
#        print("could not read")
    
    if f==None:
         f = h5py.File(filename+ extension, "w")
    
    
    if type(dict0) == dict:
        
        for i in dict0.keys():
            
            print("create group %s"%i)
            print("---")
            print(i,",",type(dict0[i]))

            if type(dict0[i]) == dict:
                print('dict')
                grp=(f.create_group(i))
                createHDF5(dict0[i],filename,f=grp)
                
            elif type(dict0[i]) == np.ndarray:
                dset=(f.create_dataset(i, data=dict0[i]))
                print("dataset created")
                
            elif (dict0[i] != None):
                dset=(f.create_dataset(i, data=dict0[i]))
                print("dataset created")
            print("---")
    return f


#############################################################
#    save parameters
#############################################################


def save_reco_dict_to_hdf(fname, reco_dict):
    '''Saves a flat dictionary to a new hdf group in given file.
    
    Parameters
    ----------
    fname : str
        hdf file name
    reco_dict : dict
        Flat dictionary
    
    Returns
    -------
    grp : str
        Name of the new data group.
    -------
    author: dscran 2020
    '''
    with h5py.File(fname, mode='a') as f:
        i = 0
        while f'reco{i:02d}' in f:
            i += 1
        for k, v in reco_dict.items():
            f[f'reco{i:02d}/{k}'] = v
    return f'reco{i:02d}'

def read_hdf(fname):
    '''
    reads the latest saved parameters in the hdf file
    INPUT:  fname: path and filename of the hdf file
    OUtPUT: image numbers, retrieved_p, retrieved_n, recon, propagation distance, phase and ROI coordinates in that order as a tuple
    -------
    author: KG 2020
    '''
    f = h5py.File(fname, 'r')
    i = 0
    while f'reco{i:02d}' in f:
        i += 1
    i -= 1
    
    nobs_numbers = f[f'reco{i:02d}/entry numbers no beamstop'].value
    sbs_numbers = f[f'reco{i:02d}/entry numbers small beamstop'].value
    lbs_numbers = f[f'reco{i:02d}/entry numbers large beamstop'].value
    retrieved_holo_p = f[f'reco{i:02d}/retrieved hologram positive helicity'].value 
    retrieved_holo_n = f[f'reco{i:02d}/retrieeved hologram negative helicity'].value 
    prop_dist = f[f'reco{i:02d}/Propagation distance'].value
    phase = f[f'reco{i:02d}/phase'].value
    roi = f[f'reco{i:02d}/ROI coordinates'].value

    return (nobs_numbers, sbs_numbers, lbs_numbers, retrieved_holo_p, retrieved_holo_n, prop_dist, phase, roi)


#############################################################
#    Ewald sphere projection
#############################################################

from scipy.interpolate import griddata

def inv_gnomonic(CCD, center=None, experimental_setup = {'ccd_dist': 18e-2, 'energy': 779.5, 'px_size' : 20e-6}, method='cubic' , mask=None):
    '''
    Projection on the Ewald sphere for close CCD images. Only gets the new positions on the new projected array and then interpolates them on a regular matrix
    Input:  CCD: far-field diffraction image
            z: camera-sample distance,
            center_y,center_x: pixels in excess we want to add to the borders by zero-padding so that the projected image has existing pixels to use
            px_size: size of CCD pixels
    Output: Output: projected image
    
    -------
    author: RB Nov2020
    '''
    
    # we have to caculate all new angles
    
    #points coordinates positions
    z=experimental_setup['ccd_dist']
    px_size=experimental_setup['px_size']
    if type(center)==type(None):
        center=np.array([CCD.shape[1]/2, CCD.shape[0]/2])


    print("center=",center, "z=",z )
    values=CCD.flatten()
    points=(np.array(np.unravel_index(np.arange(values.size), CCD.shape))).astype('float64')
    
    points[0,:]-=center[0]
    points[1,:]-=center[1]
    points*= px_size
    
    
    #points=(np.array(np.unravel_index(np.arange(values.size), CCD.shape))- CCD.shape[0]/2) * px_size

    points=points.T
        
    #now we have to calculate the new points
    points2=np.zeros(points.shape)
    points2[:,0]= z* np.sin( np.arctan( points[:,0] / np.sqrt( points[:,1] **2 + z**2 ) ) )
    points2[:,1]= z* np.sin( np.arctan( points[:,1] / np.sqrt( points[:,0] **2 + z**2 ) ) )

    
    CCD_projected = griddata(points2, values, points, method=method)
    
    CCD_projected = np.reshape(CCD_projected, CCD.shape)
    
    #makes outside from nan to zero
    CCD_projected=np.nan_to_num(CCD_projected, nan=0, posinf=0, neginf=0)
    

    return CCD_projected, points2, points, values


########
# WIDGET MASK SELECTION
########

import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
    
    
def holomask(holo, plot_max=94, RHdict={}):

    #DEFINE FUNCTION
    def add_on_button_clicked(b):
        
        x1, x2 = ax.get_xlim()
        y2, y1 = ax.get_ylim()

        # obj position
        x = fth.integer(x1 + (x2 - x1)/2)
        y = fth.integer(y1 + (y2 - y1)/2)
        #object radius
        r = fth.integer(np.maximum((x2 - x1)/2, (y2 - y1)/2))
        
        RHn=len(RHdict.keys())+1

        RHdict.update({"RH%d"%RHn: {"#":RHn,"r":r,"x":x,"y":y}})

        ax.set_xlim(0,holo.shape[1])
        ax.set_ylim(holo.shape[0],0)
        
        for i in RHdict:
            yy, xx = disk((i["y"],i["x"]),i["r"])

        
    fig, ax = plt.subplots()
    image=np.abs(fth.reconstruct(holo))
    mi,ma=np.percentile(image, (2,plot_max))
    ax.imshow(image,  vmin=mi, vmax=ma)
    

    add_button = widgets.Button(description="Add RH")
    output = widgets.Output()
    add_button.on_click(add_on_button_clicked)

    display(add_button, output)

    return RHdict


from scipy.ndimage.interpolation import shift
from scipy import signal
from numpy import unravel_index
def load_aligning(a,b, pad_factor=2):
    #pad them
    shape=np.array(a.shape)//2
    a=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(a)))
    b=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(b)))
    padding_x=(pad_factor-1)*a.shape[1]//2
    padding_y=(pad_factor-1)*a.shape[0]//2
    pad_width=((padding_x,padding_x),(padding_y,padding_y))
    a=np.pad(a, pad_width=pad_width )
    b=np.pad(b, pad_width=pad_width )
    
    #c=signal.correlate(a,b, method="fft", mode="same")
    
    c=np.conjugate(a)*b
    c=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(c)))

    center=np.array(unravel_index(c.argmax(), c.shape))
    print(center)
    center=center/pad_factor-shape
    print(center)
    return center
    