"""
Python Dictionary for CCI correlaion analysis

2021
@authors:   CK: Christopher Klose (christopher.klose@mbi-berlin.de)
            MS: Michael Schneider (michaelschneider@mbi-berlin.de)
"""

import sys, os
from os.path import join
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import itertools

from scipy.ndimage.filters import gaussian_filter
from skimage.draw import circle

#Clustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

#colormap
from matplotlib.colors import LinearSegmentedColormap
import cmap as cmap

#parula map
def parula_map():
    cm_data = cmap.parula_cmap()
    parula = LinearSegmentedColormap.from_list('parula', cm_data)
    
    return parula

def reconstruct(image):
    '''
    Reconstruct the image by fft
    -------
    author: MS 2016
    '''
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image)))

def FFT(image):
    '''
    Fourier transform
    -------
    author: CK 2021
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))


def lowpass(holo, diameter, sigma = 3, center = None):
    '''
    A smoothed circular region of the imput image is set to zero.
    
    Parameters
    ----------
    holo : array
        input hologram
    diameter: scalar
        diameter of lowpass filter in pixels
    sigma: scalar or sequence of scalars, optional
        Passed to scipy.ndimage.gaussian_filter(). Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes. Default is 3.
    center: sequence of scalars, optional
        If given, the beamstop is masked at that position, otherwise the center of the image is taken. Default is None.
    
    Returns
    -------
    masked_holo: array
        hologram with smoothed beamstop edges
    -------
    author: KG 2021
    '''
    if center is None:
        x0, y0 = [np.around(c/2) for c in holo.shape]
    else:
        x0, y0 = [np.around(c) for c in center]

    #create the beamstop mask using scikit-image's circle function
    lowpass = np.zeros(holo.shape)
    yy, xx = circle(y0, x0, diameter/2)
    lowpass[yy, xx] = 1
    #smooth the mask with a gaussion filter    
    lowpass = gaussian_filter(lowpass, sigma, mode='constant', cval=0)
    return holo*lowpass

def highpass(holo, diameter, sigma = 3, center = None):
    '''
    A smoothed circular region of the imput image is set to zero.
    
    Parameters
    ----------
    holo : array
        input hologram
    diameter: scalar
        diameter of highpass filter in pixels
    sigma: scalar or sequence of scalars, optional
        Passed to scipy.ndimage.gaussian_filter(). Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes. Default is 3.
    center: sequence of scalars, optional
        If given, the beamstop is masked at that position, otherwise the center of the image is taken. Default is None.
    
    Returns
    -------
    masked_holo: array
        hologram with smoothed beamstop edges
    -------
    author: KG 2021
    '''
    if center is None:
        x0, y0 = [c/2 for c in holo.shape]
    else:
        x0, y0 = [c for c in center]

    #create the beamstop mask using scikit-image's circle function
    lowpass = np.zeros(holo.shape)
    yy, xx = circle(y0, x0, diameter/2)
    lowpass[yy, xx] = 1
    #smooth the mask with a gaussion filter    
    lowpass = gaussian_filter(lowpass, sigma, mode='constant', cval=0)
    return holo*(1-lowpass)

def filter_reference(holo,mask,settings):
    '''
    Filter reference-induced modulations from fth holograms
    
    Parameters
    ----------
    holo : array
        input hologram
    mask: array
        (smooth) mask to crop cross correlation in Patterson map
    settings: dict
        contains parameter for cropping
    
    Returns
    -------
    holo_filtered: array
        reference-filtered "hologram"
    -------
    author: CK 2021
    '''
    
    center = settings['center']
    diameter = settings['low_dia']
    
    #Transform to Patterson map
    tmp_array = reconstruct(holo)
    
    #Crop Patterson map
    tmp_array = tmp_array*mask
    tmp_array = tmp_array[int(center[1]-diameter/2):int(center[1]+diameter/2),
                          int(center[0]-diameter/2):int(center[0]+diameter/2)]
    
    #Crop ROI of holograms
    tmp_array = FFT(tmp_array)
    tmp_array = np.real(tmp_array)
    holo_filtered = tmp_array[10:-10,10:-10]
    
    return holo_filtered


def seg_statistics(holo, Diameter, NrStd = 1, center = None):
    '''
    Creates mask that shows only value outside of a noise intervall defined by the statistics of the array
    
    Parameters
    ----------
    holo : array
        input hologram
    diameter: scalar
        diameter of highpass filter to calc noise level in outer areas
    NrStd: scalar, optional
        Multiplication factor of the standard deviation to count a pixel as noise. Default is 1.
    center: sequence of scalars, optional
        If given, the beamstop is masked at that position, otherwise the center of the image is taken. Default is None.
    
    Returns
    -------
    statistics mask: array
        bool mask of values larger than noise level
    -------
    author: CK 2021
    '''

    #Calc Statistics mask
    if center is None:
        x0, y0 = [c/2 for c in holo.shape]
    else:
        x0, y0 = [c for c in center]
    
    temp_mask = np.zeros(holo.shape)
    yy, xx = circle(y0, x0, Diameter/2)
    temp_mask[yy, xx] = 1
    
    temp = holo[temp_mask == 0]

    MEAN = np.mean(temp)
    STD  = np.std(temp)

    Statistics_mask = (np.abs(holo) >= MEAN + NrStd*STD)

    return Statistics_mask


def correlate_holograms(diff1, diff2, sum1, sum2, Statistics1, Statistics2):
    '''
    Function to determine the correlation of two holograms.
    
    Parameters
    ----------
    diff1 : array
        difference hlogram of the first data image
    diff2 : array
        difference hlogram of the second data image
    sum1: array
        sum hologram of the first data image
    sum2: array 
        sum hologram of the first data image
    
    Returns
    -------
    c_val : scalar
        correlation value of the two holograms
    c_array: array
        pixelwise correlation array of the two holograms
    -------
    author: CK 2020-2021 / KG 2021
    '''    
    # replace all zeros in sum1/sum2 with another value to avoid infinities
    sum1 = np.abs(sum1)
    sum1 = np.abs(sum2)
    
    sum1[sum1 == 0] = 1e-8
    sum2[sum2 == 0] = 1e-8
    
    #Get real part of diff
    
    
    # Combine Statistics Mask
    mask = np.logical_or(Statistics1,Statistics2)
    
    # Calc flattened holos called scattering images
    S1 = diff1*mask/np.sqrt(sum1)
    S2 = diff2*mask/np.sqrt(sum2)
   
    # normalization Factor called scattering factor
    sf = np.sqrt(np.sum(S1 * S1)*np.sum(S2 * S2))
    
    # calculate the pixelwise correlation
    c_array = S1 * S2 / sf
    
    # average correlation
    c_val = np.sum(c_array)
    
    return (c_val, c_array)


def correlation_map(diff_holo, sum_holo, statistics_mask):
    '''
    Function to determine the correlation of two holograms.
    
    Parameters
    ----------
    diff_holo : array
        array of all difference holograms
    sum_holo : array
        darray of all sum holograms, must be of the same length as diff_holo
    
    Returns
    -------
    corr_map : array
        correlation map where every image is correlatetd to each other image in the input
    -------
    author: KG 2019/2021, CK 2021
    '''
    n = diff_holo.shape[0]
    corr_map = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c_val, _ = correlate_holograms(diff_holo[i], diff_holo[j], sum_holo[i], sum_holo[j],
                                           statistics_mask[i], statistics_mask[j])
            corr_map[i][j] = c_val
    return corr_map


def reconstruct_correlation_map(frames,corr_array):
    '''
    Script Reconstruct the cluster's correlation map from the given
    cluster's 'frames' and the (large) correlation map of all
    frames.
    
    Parameters
    ----------
    frames: array
        relevant frames
    corr_array: array
        complete pair correlation map    
        
    Returns
    -------
    temp_core: array
        section of (large) correlation map defined by 'frames'
    -------
    author: CK 2021
    '''
    
    print('Reconstructing correlation map')
    
    #Reshape frame array
    frames = np.reshape(frames,frames.shape[0])
    
    #Indexing of correlation array
    temp_corr = corr_array[np.ix_(frames,frames)]
    
    print('Reconstruction finished!')
    
    return temp_corr


def create_linkage(cluster_idx,corr_array):
    '''
    calculates distance metric, linkage and feedback plots
    
    Parameters
    ----------
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    corr_array: array
        pair correlation map
        
    Returns
    -------
    tlinkage: array
        clustering linkage array
    -------
    author: CK 2021
    '''
    #get colomap
    parula = parula_map()
    
    #Calc distance metric and its squareform
    dist_metric = pdist(corr_array, metric='correlation')
    dist_metric_sq = squareform(dist_metric)
    
    #Calculate Linkage
    tlinkage = linkage(dist_metric_sq,method='average',metric='correlation')
    
    nr_cluster = 2
    temp_assignment = fcluster(tlinkage,nr_cluster,criterion='maxclust')
    
    #Output plots
    fig, _ = plt.subplots(figsize = (8,8))
    fig.suptitle(f'Cluster Index: {cluster_idx}')
    
    #Corr map
    ax1 = plt.subplot(2,2,1)
    vmi, vma = np.percentile(corr_array[corr_array != 1],[5,95])
    ax1.imshow(corr_array, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
    ax1.set_title('Correlation map')
    ax1.set_xlabel('Frame index k')
    ax1.set_ylabel('Frame index k')

    #Dist metric
    ax2 = plt.subplot(2,2,2,sharex=ax1,sharey=ax1)
    vmi, vma = np.percentile(dist_metric_sq[dist_metric_sq != 0],[1,99])
    ax2.imshow(dist_metric_sq, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
    ax2.set_title('Distance metric')
    ax2.set_xlabel('Frame index k')
    ax2.set_ylabel('Frame index k')
    plt.gca().invert_yaxis()

    #Assignment plot
    ax3 = plt.subplot(2,2,3,sharex=ax1)
    ax3.plot(temp_assignment)
    ax3.set_title('Frame assignment')
    ax3.set_xlabel('Frame index k')
    ax3.set_ylabel('State')
    ax3.set_ylim((0.5,2.5))
    ax3.set_yticks([1,2])

    #Assignment plot
    ax4 = plt.subplot(2,2,4)
    dendrogram(tlinkage, p=100, truncate_mode = 'lastp')

    plt.tight_layout()
    
    return tlinkage


def cluster_hierarchical(tlinkage,parameter,clusteringOption='maxclust'):
    '''
    calculates distance metric, linkage and feedback plots
    
    Parameters
    ----------
    tlinkage: array
        clustering tlinkage array
    parameter: scalar
        parameter of clustering option, e.g., nr of clusters
    clusteringOption: string
        criterion used in forming flat clusters
        - 'inconsistent': cluster inconsistency threshold
        - 'maxcluster': number of total clusters
        
    Returns
    -------
    cluster_assignment: array
        assignment of frames to cluster
    -------
    author: CK 2021
    '''
    
    #Options
    if clusteringOption == 'maxclust':
        criterion_ = 'maxclust'
    elif clusteringOption == 'inconsistent':
        criterion_ = 'inconsistent'
    else:
        print('Error: clustering option not valid!')

    #Get cluster
    cluster_assignment = fcluster(tlinkage,parameter,criterion=criterion_)

    #Feedback
    nr = np.unique(cluster_assignment).shape[0]
    print(f'{nr} clusters were constructed!')
    
    return cluster_assignment


def clustering_feedback(cluster_idx,nr,corr_array_large,corr_array_small,dist_metric_sq,tlinkage):
    '''
    calculates distance metric, linkage and feedback plots
    
    Parameters
    ----------
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    nr: scalar
        index of subcluster
    corr_array_large: array
        initial pair correlation map
    corr_array_small: array
        pair correlation map of new subcluster
    dist_metric_sq: array
        distance metric of pair correlation map in square format
    tlinkage: array
        clustering linkage array
        
    Returns
    -------
    fig with plots
    -------
    author: CK 2021
    '''
    
    #get colomap
    parula = parula_map()
    
    #Output plots
    fig, _ = plt.subplots(figsize = (8,8))
    fig.suptitle(f'Cluster Index: {cluster_idx}-{nr}')
    
    #section of Initial Corr map
    ax1 = plt.subplot(2,2,1)
    vmi, vma = np.percentile(corr_array_large[corr_array_large != 1],[5,95])
    ax1.imshow(corr_array_large, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
    ax1.set_title('Section initial correlation map')
    ax1.set_xlabel('Frame index k')
    ax1.set_ylabel('Frame index k')
    plt.gca().invert_yaxis()
    
    #section of Initial Corr map
    ax2 = plt.subplot(2,2,2)
    vmi, vma = np.percentile(corr_array_small[corr_array_small != 1],[5,95])
    ax2.imshow(corr_array_small, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
    ax2.set_title('New correlation map')
    ax2.set_xlabel('Frame index k')
    ax2.set_ylabel('Frame index k')

    #Dist metric
    ax3 = plt.subplot(2,2,3,sharex=ax2,sharey=ax2)
    vmi, vma = np.percentile(dist_metric_sq[dist_metric_sq != 0],[1,99])
    ax3.imshow(dist_metric_sq, vmin = vmi, vmax = vma, cmap = parula, aspect="auto")
    ax3.set_title('Distance metric')
    ax3.set_xlabel('Frame index k')
    ax3.set_ylabel('Frame index k')
    plt.gca().invert_yaxis()

    #Assignment plot
    ax4 = plt.subplot(2,2,4)
    dendrogram(tlinkage, p=150, truncate_mode = 'lastp')

    plt.tight_layout()
    return


def process_cluster(cluster,cluster_idx,corr_array,cluster_assignment,save=False):
    '''
    processes a given cluster assignment and adds new subclusters to 'cluster'-list
    
    Parameters
    ----------
    cluster: list of dictionaries
        stores relevant data of clusters, e.g., assigned frames
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    corr_array: array
        pair correlation map
    cluster_assignment: array
        assignment of frames to cluster
    save: bool
        save new subclusters in "cluster"-list and delete current cluster from list
    
    Returns
    -------
    cluster: list of dicts
        updated "cluster"-list
    -------
    author: CK 2021
    '''
    
    length = len(cluster)
    
    #Get initial frames in cluster
    frames = cluster[cluster_idx]['Cluster_Frames']
    frames = np.reshape(frames,frames.shape[0])
    
    #Get nr of new subclusters
    nr = np.unique(cluster_assignment)
    
    #Vary subclusters
    for ii in nr:
        print(f'Creating sub-cluster: {cluster_idx}-{ii}')
    
        #Get assignment
        tmp_assignment = np.argwhere(cluster_assignment == ii)
        tmp_assignment = np.reshape(tmp_assignment,tmp_assignment.shape[0])
        
        #Get subcluster correlation array
        tmp_corr_small = corr_array[np.ix_(tmp_assignment,tmp_assignment)]
        
        #Create mask which selects the section of the correlation that is assigned to sub-cluster ii
        #tmp_mask = np.zeros([cluster_assignment.shape[0],cluster_assignment.shape[0]])
        #tmp_mask[np.ix_(tmp_assignment,tmp_assignment)] = corr_array[np.ix_(tmp_assignment,tmp_assignment)]
        tmp_corr_large = np.zeros([cluster_assignment.shape[0],cluster_assignment.shape[0]])
        tmp_corr_large[np.ix_(tmp_assignment,tmp_assignment)] = corr_array[np.ix_(tmp_assignment,tmp_assignment)]
        
        if len(tmp_assignment) > 1:
            #Calc sub-cluster distance metric
            dist_metric = pdist(tmp_corr_small, metric='correlation')
            dist_metric_sq = squareform(dist_metric)

            #Calculate Linkage
            tlinkage = linkage(dist_metric_sq,method='average',metric='correlation')

            #Plots
            clustering_feedback(cluster_idx,ii,tmp_corr_large,tmp_corr_small,dist_metric_sq,tlinkage)
        
        #Save new cluster
        if save == True:
            print(f'Saving subcluster {cluster_idx}-{ii} as new cluster {length + ii}')
            cluster.append({"Cluster_Nr": length + ii,"Cluster_Frames": frames[np.ix_(tmp_assignment)]})
    
    #Del old cluster from 'cluster'-list
    if save == True:
        cluster[cluster_idx] = {}
                            
    return cluster