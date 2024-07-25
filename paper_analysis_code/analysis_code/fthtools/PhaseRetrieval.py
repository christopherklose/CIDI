
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
import scipy as sp
from scipy.fftpack import fft2, ifftshift, fftshift,ifft2
import scipy.io
from scipy.stats import linregress
# import fthcore as fth
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets
from skimage.draw import disk
import h5py
import math
from tqdm.auto import trange
# import cupy as cp
# import cupyx as cx #.scipy.ndimage.convolve1d
# import gc
#from scipy.ndimage import gaussian_filter
# import cupyx.scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label

#############################################################
#       PHASE RETRIEVAL FUNCTIONS
#############################################################

def PhaseRtrv_init(diffract,mask, smth_func=1,Nit=500,beta_zero=0.5, beta_mode='const',gamma=None, Phase=0, seed=False, bsmask=0,SW_sigma_list=0, SW_thr_list=0):
    '''
    Initialization for terative phase retrieval function
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            gamma: starting guess for MOI
            RL_freq: number of steps between a gamma update and the next
            RL_it: number of steps for every gamma update
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
 
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''    
    #set parameters and BSmask
    (l,n) = diffract.shape
    Error_diffr_list=cp.zeros(0)
    Error_supp_list=cp.zeros(0)
    Error_supp=0

    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
        
        
    #prepare SW_sigma_list and SW_thr_list
    #....
          
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    #print("using phase given")

    if type(gamma)!=type(None):
        gamma=np.abs(np.fft.fftshift(gamma)) #####
        gamma_cp=cp.asarray(gamma)
        gamma_cp/=cp.sum((gamma_cp))
    else:
        gamma_cp=None
        
    #guess = (1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
    
    if type(Phase)==int:
        Phase=np.exp(1j * np.random.rand(l,n)*np.pi*2)
        Phase=(1-bsmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    #guess= Phase.copy()
    guess = (1-bsmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    

  
    #previous result
    prev = None
    
    #shift everything to the corner
    bsmask=np.fft.fftshift(bsmask)
    guess=np.fft.fftshift(guess)
    mask=np.fft.fftshift(mask)
    diffract=np.fft.fftshift(diffract)
    
    BSmask_cp=cp.asarray(bsmask)
    guess_cp=cp.asarray(guess)
    mask_cp=cp.asarray(mask)
    diffract_cp=cp.asarray(diffract)
    if type(Phase)==int:
        smth_func_cp=smth_func
    else:
        smth_func_cp=cp.asarray(smth_func)
    

    if type(gamma)!=type(None):
        convolved=cp.fft.ifft2(cp.fft.fft2(cp.abs(guess_cp)**2) * cp.fft.fft2(gamma_cp))
        prev=cp.fft.fft2((1-BSmask_cp) *diffract_cp/cp.sqrt(convolved)* guess_cp + guess_cp * BSmask_cp)
    else:
        convolved=None
        prev=cp.fft.fft2((1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp)

    return Error_diffr_list,Error_supp_list, Beta, gamma_cp, guess, BSmask_cp, guess_cp, mask_cp, diffract_cp, smth_func_cp, convolved, prev,SW_sigma_list, SW_thr_list


def PhaseRtrv_init_CPU(diffract,mask, smth_func=1,Nit=500,beta_zero=0.5, beta_mode='const',gamma=None, Phase=0, seed=False, bsmask=0,SW_sigma_list=0, SW_thr_list=0):
    '''
    Initialization for terative phase retrieval function
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            gamma: starting guess for MOI
            RL_freq: number of steps between a gamma update and the next
            RL_it: number of steps for every gamma update
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
 
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''    
    #set parameters and BSmask
    (l,n) = diffract.shape
    Error_diffr_list=np.zeros(0)
    Error_supp_list=np.zeros(0)
    Error_supp=0

    
    #prepare beta function
    step=np.arange(Nit)
    if type(beta_mode)==np.ndarray:
        Beta=beta_mode
    elif beta_mode=='const':
        Beta=beta_zero*np.ones(Nit)
    elif beta_mode=='arctan':
        Beta=beta_zero+(0.5-np.arctan((step-np.minimum(Nit/2,700))/(0.15*Nit))/(np.pi))*(0.98-beta_zero)
        
        
    #prepare SW_sigma_list and SW_thr_list
    #....
          
    if seed==True:
        np.random.seed(0)
      
    #set initial Phase guess
    #print("using phase given")

    if type(gamma)!=type(None):
        gamma=np.abs(np.fft.fftshift(gamma)) #####
        gamma_cp=np.array(gamma)
        gamma_cp/=np.sum(gamma_cp)
    else:
        gamma_cp=None
        
    #guess = (1-BSmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*BSmask
    
    if type(Phase)==int:
        Phase=np.exp(1j * np.random.rand(l,n)*np.pi*2)
        Phase=(1-bsmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    #guess= Phase.copy()
    guess = (1-bsmask)*diffract * np.exp(1j * np.angle(Phase))+ Phase*bsmask
    

  
    #previous result
    prev = None
    
    #shift everything to the corner
    bsmask=sp.fft.fftshift(bsmask)
    guess=sp.fft.fftshift(guess)
    mask=sp.fft.fftshift(mask)
    diffract=sp.fft.fftshift(diffract)
    
    BSmask_cp=np.array(bsmask)
    guess_cp=np.array(guess)
    mask_cp=np.array(mask)
    diffract_cp=np.array(diffract)
    if type(Phase)==int:
        smth_func_cp=smth_func
    else:
        smth_func_cp=np.array(smth_func)
    

    if type(gamma)!=type(None):
        convolved=sp.fft.ifft2(sp.fft.fft2(np.abs(guess_cp)**2) * sp.fft.fft2(gamma_cp))
        prev=sp.fft.fft2((1-BSmask_cp) *diffract_cp/np.sqrt(convolved)* guess_cp + guess_cp * BSmask_cp)
    else:
        convolved=None
        prev=sp.fft.fft2((1-BSmask_cp) *diffract_cp* sp.exp(1j * np.angle(guess_cp)) + guess_cp*BSmask_cp)

    return Error_diffr_list,Error_supp_list, Beta, gamma_cp, guess, BSmask_cp, guess_cp, mask_cp, diffract_cp, smth_func_cp, convolved, prev,SW_sigma_list, SW_thr_list




def ShrinkWrap(inv,mask,SW_sigma=1, SW_thr=0.01):
    '''
    Shrink-wrap, with GPU acceleration
    INPUT:  inv: real_space reconstruction
            SW_sigma: sigma of the gaussian
            SW_thr: threshold used for the definition with respect to the maximum   
    OUTPUT:  new support mask
     --------
    author: RB 2022
    '''
    
    mask_labeled,nb_labels=label(mask>=.5)
    mask_new=np.zeros(mask.shape)
    #blurring real space image
    blurred=gaussian_filter(np.abs(inv),sigma=SW_sigma)
    
    #thresholding it
    labeled=False
    if labeled:
        for i in np.arange(1,nb_labels+1):
            mask_new+=1*(mask_labeled==i)*(blurred>=(SW_thr*np.amax(blurred[mask_labeled==i])))
    else:
        mask_new=1*(blurred>=(SW_thr*np.amax(blurred[mask==1])))
        
        
    # only where the mask was already there
    mask_new = mask_new * mask
    

    # fill holes
    #mask_new = ndi.binary_fill_holes(mask_new)
    #fig,ax=plt.subplots()
    #ax.imshow(cp.asnumpy(mask_new))
    
    return mask_new
    

#################
def PhaseRtrv_GPU(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,seed=False,
       plot_every=20, bsmask=0,real_object=False,average_img=10, Fourier_last=True, SW_freq=2e9,SW_sigma_list=0, SW_thr_list=0):
    
    '''
    Iterative phase retrieval function, with GPU acceleration
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT:  retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    alpha=0.4
    (l,n) = diffract.shape
    Error_diffr_list,Error_supp_list, Beta, gamma_cp, guess, BSmask_cp, guess_cp, mask_cp, diffract_cp, smth_func_cp, convolved, prev,SW_sigma_list, SW_thr_list = PhaseRtrv_init(diffract,mask, smth_func=1,Nit=Nit,beta_zero=beta_zero, beta_mode=beta_mode,gamma=None, Phase=Phase, seed=False, bsmask=bsmask,SW_sigma_list=SW_sigma_list, SW_thr_list=SW_thr_list)
    first_instance_error=False
    
    for s in range(0,Nit):

        beta=Beta[s]

        ###REAL SPACE###
        inv=cp.fft.fft2( guess_cp*((1-BSmask_cp) *diffract_cp/np.abs(guess_cp) + BSmask_cp))
        
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='mine':
            inv += beta*(prev - 2*inv)*(1-mask_cp)
        elif mode=='RAAR':
            inv += beta*(prev - 2*inv)*(1-mask_cp) * (2*inv-prev<0)
        elif mode=='HIOs':
            inv +=  (1-mask_cp)*(prev - (beta+1) * inv)
        elif mode=='HIO':
            inv += (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*(cp.real(inv)<0)
        elif mode=='OSS':
            inv += (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*(cp.real(inv)<0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
            smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
            inv = (inv*mask_cp + (1-mask_cp)*smoothed)
        elif mode=='CHIO':
            inv= (prev-beta*inv) + mask_cp*(cp.real(inv-alpha*prev)>=0)*(-prev+(beta+1)*inv)+ (cp.real(-inv+alpha*prev)>=0)*(cp.real(inv)>=0)*((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            inv +=  (1-mask_cp)*(prev - (beta+1) * inv)+ mask_cp*(prev - (beta+1) * inv)*(cp.real(prev-(beta-3)*inv)>0)

        #SHRINK_WRAP 
        if (s>0) and ((s%SW_freq)==0):
            mask_cp=ShrinkWrap(inv,mask_cp,SW_sigma_list[s], SW_thr_list[s])
            
            
        prev=cp.copy(inv)

        
        ### FOURIER SPACE ### 
        guess_cp=cp.fft.ifft2(inv)
                
        #COMPUTE ERRORS
        if s<=2 or s % plot_every == 0:
            
            Error_diffr = Error_diffract(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
            Error_diffr_list=cp.append(Error_diffr_list,Error_diffr)
            #print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)
            
        elif s>=2 and s>= Nit-average_img*2:
            
            if first_instance_error==False:
                Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
                Best_error=cp.zeros(average_img)+1e10
                first_instance_error=True
            
            #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
            Error_diffr = Error_diffract(cp.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
            Error_diffr_list=cp.append(Error_diffr_list,Error_diffr)
            
            if Error_diffr<=cp.amax(Best_error):
                j=cp.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j] = cp.copy(Error_diffr)
                    Best_guess[j,:,:]=cp.copy(guess_cp)


    #sum best guess images
    guess_cp=cp.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp* cp.exp(1j * cp.angle(guess_cp)) + guess_cp*BSmask_cp
    
    guess=cp.asnumpy(guess_cp)
    Error_diffr_list=cp.asnumpy(Error_diffr_list)
    mask=np.fft.ifftshift(cp.asnumpy(mask_cp))

    #return final image
    return (np.fft.ifftshift(guess)), Error_diffr_list, Error_supp_list, mask
    

def PhaseRtrv_CPU(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const', Phase=0,seed=False,
       plot_every=20, bsmask=0,real_object=False,average_img=10, Fourier_last=True, SW_freq=2e9,SW_sigma_list=0, SW_thr_list=0, progressbar=False):
    
    '''
    Iterative phase retrieval function, with GPU acceleration
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
            BS: [centery,centerx,radius] of BeamStopper
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT:  retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''

 
    alpha=0.4
    (l,n) = diffract.shape
    Error_diffr_list,Error_supp_list, Beta, gamma_cp, guess, BSmask_cp, guess_cp, mask_cp, diffract_cp, smth_func_cp, convolved, prev,SW_sigma_list, SW_thr_list = PhaseRtrv_init_CPU(diffract,mask, smth_func=1,Nit=Nit,beta_zero=beta_zero, beta_mode=beta_mode,gamma=None, Phase=Phase, seed=False, bsmask=bsmask,SW_sigma_list=SW_sigma_list, SW_thr_list=SW_thr_list)
    first_instance_error=False
    
    _range = trange if progressbar else range
    
    for s in _range(0, Nit):

        beta=Beta[s]

        ###REAL SPACE###
        inv=sp.fft.fft2(guess_cp * (
            (1 - BSmask_cp) * diffract_cp / (1e-9 + np.abs(guess_cp)) + BSmask_cp
        ))
        
        #support Projection
        if mode=='ER':
            inv=inv*mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='mine':
            inv += beta*(prev - 2*inv)*(1-mask_cp)
        elif mode=='RAAR':
            inv += beta*(prev - 2*inv)*(1-mask_cp) * (2*inv-prev<0)
        elif mode=='HIOs':
            inv +=  (1-mask_cp)*(prev - (beta+1) * inv)
        elif mode=='HIO':
            inv += (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*(np.real(inv)<0)
        elif mode=='OSS':
            inv += (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)*(np.real(inv)<0)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* np.floor(s/Nit*10)/10
            smoothed= sp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * sp.fft.fft2(inv))          
            inv = (inv*mask_cp + (1-mask_cp)*smoothed)
        elif mode=='CHIO':
            inv= (prev-beta*inv) + mask_cp*(np.real(inv-alpha*prev)>=0)*(-prev+(beta+1)*inv)+ (np.real(-inv+alpha*prev)>=0)*(np.real(inv)>=0)*((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            inv +=  (1-mask_cp)*(prev - (beta+1) * inv)+ mask_cp*(prev - (beta+1) * inv)*(np.real(prev-(beta-3)*inv)>0)

        #SHRINK_WRAP 
        if (s>0) and ((s%SW_freq)==0):
            mask_cp=ShrinkWrap(inv,mask_cp,SW_sigma_list[s], SW_thr_list[s])
            
            
        prev=np.copy(inv)

        
        ### FOURIER SPACE ### 
        guess_cp=sp.fft.ifft2(inv)
                
        #COMPUTE ERRORS
        if s<=2 or s % plot_every == 0:
            
            Error_diffr = Error_diffract(np.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
            Error_diffr_list=np.append(Error_diffr_list,Error_diffr)
            #print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)
            
        elif s>=2 and s>= Nit-average_img*2:
            
            if first_instance_error==False:
                Best_guess=np.zeros((average_img,l,n),dtype = np.complex64)
                Best_error=np.zeros(average_img)+1e10
                first_instance_error=True
            
            #compute error to see if image has to end up among the best. Add in Best_guess array to sum them up at the end
            Error_diffr = Error_diffract(np.abs(guess_cp)*(1-BSmask_cp), diffract_cp*(1-BSmask_cp))
            Error_diffr_list=np.append(Error_diffr_list,Error_diffr)
            
            if Error_diffr<=np.amax(Best_error):
                j=np.argmax(Best_error)
                if Error_diffr<Best_error[j]:
                    Best_error[j] = np.copy(Error_diffr)
                    Best_guess[j,:,:]=np.copy(guess_cp)


    #sum best guess images
    guess_cp=np.sum(Best_guess,axis=0)/average_img
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp* np.exp(1j * np.angle(guess_cp)) + guess_cp*BSmask_cp
    
    guess=np.array(guess_cp)
    Error_diffr_list=np.array(Error_diffr_list)
    mask=np.fft.ifftshift(np.array(mask_cp))

    #return final image
    return (np.fft.ifftshift(guess)), Error_diffr_list, Error_supp_list, mask


def PhaseRtrv_with_RL_CPU(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const',gamma=None, RL_freq=25, RL_it=20, Phase=0,seed=False,
       plot_every=20, bsmask=0, real_object=False, average_img=10, Fourier_last=True, smth_func=1, SW_freq=200000,SW_sigma_list=0, SW_thr_list=0):
    
    '''
    Iterative phase retrieval function, with GPU acceleration and Richardson Lucy algorithm
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            gamma: starting guess for MOI
            RL_freq: number of steps between a gamma update and the next
            RL_it: number of steps for every gamma update
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
 
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''
    alpha=0.4
    (l,n) = diffract.shape
    Error_diffr_list,Error_supp_list, Beta, gamma_cp, guess, BSmask_cp, guess_cp, mask_cp, diffract_cp, smth_func_cp, convolved, prev,SW_sigma_list, SW_thr_list = PhaseRtrv_init_CPU(diffract,mask, smth_func,Nit=Nit,beta_zero=beta_zero, beta_mode=beta_mode,gamma=gamma, Phase=Phase, seed=False, bsmask=bsmask,SW_sigma_list=SW_sigma_list, SW_thr_list=SW_thr_list)
    first_instance_error=False

    for s in trange(0,Nit):
        beta=Beta[s]
        ### MAGNITUDE CONSTRAINT -> REAL SPACE ###
        inv = sp.fft.fft2( ((1-BSmask_cp) *diffract_cp/np.sqrt(convolved) + BSmask_cp)* guess_cp )


            
        ### SUPPORT CONSTRAINT  support Projection ###
        if mode=='ER':
            inv*=mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='mine':
            inv += beta*(prev - 2*inv)*(1-mask_cp)
        elif mode=='RAAR':
            inv += beta*(prev - 2*inv)*(1-mask_cp) * (2*inv-prev<0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        elif mode=='HIO':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* np.floor(s/Nit*10)/10
            smoothed= sp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * sp.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*(np.real(inv-alpha*prev)>=0)*(-prev+(beta+1)*inv)+ (np.real(-inv+alpha*prev)>=0)*(np.real(inv)>=0)*((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)+ mask_cp*(prev - (beta+1) * inv)*(np.real(prev-(beta-3)*inv)>0)
        
        #SHRINK_WRAP
        if (s>0) and ((s%SW_freq)==0):
            mask_cp=ShrinkWrap(inv,mask_cp,SW_sigma_list[s], SW_thr_list[s])
            
            
        prev=np.copy(inv)
        
        ### -> FOURIER SPACE ### 
        new_guess=sp.fft.ifft2(inv)

        #RL algorithm
        if s>2 and (s%RL_freq==0):
            
            #convolved=FFTConvolve(cp.abs(new_guess)**2,gamma_cp)
            convolved=sp.fft.ifft2(sp.fft.fft2(np.abs(new_guess)**2) * sp.fft.fft2(gamma_cp))
            Idelta=2*np.abs(new_guess)**2-np.abs(guess_cp)**2
            I_exp=(1-BSmask_cp) *np.abs(diffract_cp)**2 + convolved * BSmask_cp
            gamma_cp = RL( Idelta=Idelta,  Iexp = I_exp , gamma_cp=gamma_cp, RL_it=RL_it, smth_func=smth_func_cp)

            
        guess_cp = new_guess.copy()
        convolved=sp.fft.ifft2(sp.fft.fft2(np.abs(guess_cp)**2) * sp.fft.fft2(gamma_cp))
        #FFTConvolve(cp.abs(guess_cp)**2,gamma_cp)
        
            
        if s<=2 or s % plot_every == 0:
            Error_diffr = Error_diffract_cp( (1-BSmask_cp) * np.abs(diffract_cp)**2,  (1-BSmask_cp) * convolved)
            Error_diffr_list=np.append(Error_diffr_list,Error_diffr)
            #print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)
            
        elif s>=2 and s>= Nit-average_img*2:
            
            if first_instance_error==False:
                Best_guess=np.zeros((average_img,l,n),dtype = 'complex_')
                Best_gamma=np.zeros((average_img,l,n),dtype = 'complex_')
                Best_error=np.zeros(average_img)+1e10
                first_instance_error=True
                
            #compute error to see if image has to end up among the best.
            #Add in Best_guess array to sum them up at the end
            
            Error_diffr = Error_diffract_cp( (1-BSmask_cp) * np.abs(diffract_cp)**2,  (1-BSmask_cp) * convolved)
            Error_diffr_list=np.append(Error_diffr_list,Error_diffr)
            
            if Error_diffr<=np.amax(Best_error):
                j=np.argmax(Best_error)
                #if Error_diffr<Best_error[j]:
                Best_error[j] = np.copy(Error_diffr)
                Best_guess[j,:,:]=np.copy(guess_cp)
                Best_gamma[j,:,:]=np.copy(gamma_cp)
                    

    #sum best guess images dividing them for the number of items in Best_guess that are different from 0
    guess_cp=np.sum(Best_guess,axis=0)/np.sum(Best_error!=0)
    gamma_cp=np.sum(Best_gamma,axis=0)/np.sum(Best_error!=0)
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp/np.sqrt(FFTConvolve(np.abs(guess_cp)**2,gamma_cp))* guess_cp + guess_cp * BSmask_cp

    guess=guess_cp
    gamma = gamma_cp
    Error_diffr_list=Error_diffr_list
    mask=sp.fft.ifftshift(mask_cp)
    
    return np.fft.ifftshift(guess), Error_diffr_list, Error_supp_list, np.fft.ifftshift(gamma),mask
    
    

    
def PhaseRtrv_with_RL(diffract,mask,mode='ER',Nit=500,beta_zero=0.5, beta_mode='const',gamma=None, RL_freq=25, RL_it=20, Phase=0,seed=False,
       plot_every=20, bsmask=0, real_object=False, average_img=10, Fourier_last=True, smth_func=1, SW_freq=200000,SW_sigma_list=0, SW_thr_list=0):
    
    '''
    Iterative phase retrieval function, with GPU acceleration and Richardson Lucy algorithm
    INPUT:  diffract: far field hologram data
            mask: support matrix, defines where the retrieved reconstruction is supposed to be zeroa
            mode: string defining the algorithm to use (ER, RAAR, HIO, CHIO, OSS, HPR)
            Nit: number of steps
            beta_zero: starting value of beta
            beta_mode: way to evolve the beta parameter (const, arctan, exp, linear_to_beta_zero, linear_to_1)
            gamma: starting guess for MOI
            RL_freq: number of steps between a gamma update and the next
            RL_it: number of steps for every gamma update
            Phase: initial image to start from, if Phase=0 it's going to be a random start
            seed: Boolean, if True, the starting value will be random but always using the same seed for more reproducible retrieved images
            plot_every: how often you plot data during the retrieval process
 
            bsmask: binary matrix used to mask camera artifacts. where it is 1, the pixel will be left floating during the phase retrieval process
            real_object: Boolean, if True the image is considered to be real
            average_img: number of images with a local minum Error on the diffraction pattern that will be summed to get the final image
            
            
    OUTPUT: retrieved diffraction pattern, list of Error on far-field data, list of Errors on support
    
     --------
    author: RB 2020
    '''
    alpha=0.4
    (l,n) = diffract.shape
    Error_diffr_list,Error_supp_list, Beta, gamma_cp, guess, BSmask_cp, guess_cp, mask_cp, diffract_cp, smth_func_cp, convolved, prev,SW_sigma_list, SW_thr_list = PhaseRtrv_init(diffract,mask, smth_func,Nit=Nit,beta_zero=beta_zero, beta_mode=beta_mode,gamma=gamma, Phase=Phase, seed=False, bsmask=bsmask,SW_sigma_list=SW_sigma_list, SW_thr_list=SW_thr_list)
    first_instance_error=False

    for s in range(0,Nit):
        beta=Beta[s]
        ### MAGNITUDE CONSTRAINT -> REAL SPACE ###
        inv = cp.fft.fft2( ((1-BSmask_cp) *diffract_cp/cp.sqrt(convolved) + BSmask_cp)* guess_cp )


            
        ### SUPPORT CONSTRAINT  support Projection ###
        if mode=='ER':
            inv*=mask_cp
        elif mode=='SF':
            inv=inv*(2*mask_cp-1)
        elif mode=='mine':
            inv += beta*(prev - 2*inv)*(1-mask_cp)
        elif mode=='RAAR':
            inv += beta*(prev - 2*inv)*(1-mask_cp) * (2*inv-prev<0)
        elif mode=='HIOs':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)
        elif mode=='HIO':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)
        elif mode=='OSS':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv) + mask_cp*(prev - (beta+1) * inv)
            #smooth region outside support for smoothing
            alpha= l - (l-1/l)* cp.floor(s/Nit*10)/10
            smoothed= cp.fft.ifft2( W(inv.shape[0],inv.shape[1],alpha) * cp.fft.fft2(inv))          
            inv= mask_cp*inv + (1-mask_cp)*smoothed
        elif mode=='CHIO':
            alpha=0.4
            inv= (prev-beta*inv) + mask_cp*(cp.real(inv-alpha*prev)>=0)*(-prev+(beta+1)*inv)+ (cp.real(-inv+alpha*prev)>=0)*(cp.real(inv)>=0)*((beta-(1-alpha)/alpha)*inv)
        elif mode=='HPR':
            inv =  inv + (1-mask_cp)*(prev - (beta+1) * inv)+ mask_cp*(prev - (beta+1) * inv)*(cp.real(prev-(beta-3)*inv)>0)
        
        #SHRINK_WRAP
        if (s>0) and ((s%SW_freq)==0):
            mask_cp=ShrinkWrap(inv,mask_cp,SW_sigma_list[s], SW_thr_list[s])
            
            
        prev=cp.copy(inv)
        
        ### -> FOURIER SPACE ### 
        new_guess=cp.fft.ifft2(inv)

        #RL algorithm
        if s>2 and (s%RL_freq==0):
            
            #convolved=FFTConvolve(cp.abs(new_guess)**2,gamma_cp)
            convolved=cp.fft.ifft2(cp.fft.fft2(cp.abs(new_guess)**2) * cp.fft.fft2(gamma_cp))
            Idelta=2*cp.abs(new_guess)**2-cp.abs(guess_cp)**2
            I_exp=(1-BSmask_cp) *cp.abs(diffract_cp)**2 + convolved * BSmask_cp
            gamma_cp = RL( Idelta=Idelta,  Iexp = I_exp , gamma_cp=gamma_cp, RL_it=RL_it, smth_func=smth_func_cp)

            
        guess_cp = new_guess.copy()
        convolved=cp.fft.ifft2(cp.fft.fft2(cp.abs(guess_cp)**2) * cp.fft.fft2(gamma_cp))
        #FFTConvolve(cp.abs(guess_cp)**2,gamma_cp)
        
            
        if s<=2 or s % plot_every == 0:
            Error_diffr = Error_diffract_cp( (1-BSmask_cp) * cp.abs(diffract_cp)**2,  (1-BSmask_cp) * convolved)
            Error_diffr_list=cp.append(Error_diffr_list,Error_diffr)
            #print('#',s,'   beta=',beta,'   Error_diffr=',Error_diffr, '   Error_supp=',Error_supp)
            
        elif s>=2 and s>= Nit-average_img*2:
            
            if first_instance_error==False:
                Best_guess=cp.zeros((average_img,l,n),dtype = 'complex_')
                Best_gamma=cp.zeros((average_img,l,n),dtype = 'complex_')
                Best_error=cp.zeros(average_img)+1e10
                first_instance_error=True
                
            #compute error to see if image has to end up among the best.
            #Add in Best_guess array to sum them up at the end
            
            Error_diffr = Error_diffract_cp( (1-BSmask_cp) * cp.abs(diffract_cp)**2,  (1-BSmask_cp) * convolved)
            Error_diffr_list=cp.append(Error_diffr_list,Error_diffr)
            
            if Error_diffr<=cp.amax(Best_error):
                j=cp.argmax(Best_error)
                #if Error_diffr<Best_error[j]:
                Best_error[j] = cp.copy(Error_diffr)
                Best_guess[j,:,:]=cp.copy(guess_cp)
                Best_gamma[j,:,:]=cp.copy(gamma_cp)
                    

    #sum best guess images dividing them for the number of items in Best_guess that are different from 0
    guess_cp=cp.sum(Best_guess,axis=0)/cp.sum(Best_error!=0)
    gamma_cp=cp.sum(Best_gamma,axis=0)/cp.sum(Best_error!=0)
    
    
    #APPLY FOURIER COSTRAINT ONE LAST TIME
    if Fourier_last==True:
        guess_cp = (1-BSmask_cp) *diffract_cp/cp.sqrt(FFTConvolve(cp.abs(guess_cp)**2,gamma_cp))* guess_cp + guess_cp * BSmask_cp

    guess=cp.asnumpy(guess_cp)
    gamma = cp.asnumpy(gamma_cp)
    Error_diffr_list=cp.asnumpy(Error_diffr_list)
    mask=np.fft.ifftshift(cp.asnumpy(mask_cp))
    
    return np.fft.ifftshift(guess), Error_diffr_list, Error_supp_list, np.fft.ifftshift(gamma),mask
    
    
    
#########


def RL2(Idelta, Iexp, gamma_cp, RL_it, smth_func=1):
    '''
    Iteration cycle for Richardson Lucy algorithm
    
    --------
    author: RB 2020
    '''

    Id_1 = cp.fft.fft2(Idelta[::-1,::-1])
    Id   = cp.fft.fft2(Idelta)
    for l in range(RL_it):
        
        gamma_cp = cp.abs(gamma_cp*cp.fft.ifft2(Id_1*cp.abs(cp.fft.fft2(Iexp/cp.abs((cp.fft.ifft2(Id*cp.fft.fft2(gamma_cp))))))))
        gamma_cp*=smth_func
        gamma_cp/=cp.nansum((gamma_cp))
        
    gamma_cp/= cp.nansum(gamma_cp)

    return cp.abs(gamma_cp)

def RL3(Idelta, Iexp, gamma_cp, RL_it, smth_func=1):
    '''
    Iteration cycle for Richardson Lucy algorithm
    
    --------
    author: RB 2020
    '''

    Id_1 = cp.fft.fft2(Idelta[::-1,::-1])
    Id   = cp.fft.fft2(Idelta)
    
    
    for l in range(RL_it):
        
        gamma_cp = cp.abs(gamma_cp*cp.fft.ifft2(Id_1*(cp.fft.fft2(Iexp/((cp.fft.ifft2(Id*cp.fft.fft2(gamma_cp))))))))

        gamma_cp[smth_func<=1e-5]=np.average(gamma_cp[smth_func<=1e-5])
        gamma_cp[smth_func>1e-5]=(gamma_cp*smth_func)[smth_func>1e-5]
        gamma_cp/=cp.nansum((gamma_cp))
        
    gamma_cp/= cp.nansum(gamma_cp)

    return cp.abs(gamma_cp)

def RL(Idelta, Iexp, gamma_cp, RL_it, smth_func=1):
    '''
    Iteration cycle for Richardson Lucy algorithm
    
    --------
    author: RB 2020
    '''

    Id_1 = sp.fft.fft2(Idelta[::-1,::-1])
    Id   = sp.fft.fft2(Idelta)
    
    #fig,ax=plt.subplots(1,2)
    
    #ax[0].plot(cp.asnumpy(gamma_cp[0,:]), ".-", label="start")
    
    
    for l in range(RL_it):
        
        Denom=((sp.fft.ifft2((sp.fft.fft2(gamma_cp)*Id))).real)
        
        
        #ax[1].plot(cp.asnumpy(Denom[0,:]), ".-", label="%d"%l)
        
        Denom[Denom<1]=1e10
        
        gamma_cp = np.abs(gamma_cp*sp.fft.ifft2(Id_1*(sp.fft.fft2(Iexp/Denom))))

        gamma_cp[smth_func<=1e-5]=np.average(gamma_cp[smth_func<=1e-5])
        gamma_cp[smth_func>1e-5]=(gamma_cp*smth_func)[smth_func>1e-5]
        gamma_cp/=np.nansum((gamma_cp))
        
        #ax[0].plot(cp.asnumpy(gamma_cp[0,:]), ".-", label="%d"%l)
        
    gamma_cp/= np.nansum(gamma_cp)
    
    """
    ax[0].legend()
    ax[1].legend()
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    """

    return np.abs(gamma_cp)



def RL_old(Idelta, Iexp, gamma_cp, RL_it):
    '''
    Iteration cycle for Richardson Lucy algorithm
    
    --------
    author: RB 2020
    '''
    
    for l in range(RL_it):
        
        gamma_cp = (gamma_cp * (FFTConvolve(Idelta[::-1,::-1], Iexp/(FFTConvolve(Idelta,gamma_cp)))))
        gamma_cp/=cp.sum((gamma_cp))
        
    gamma_cp/= cp.nansum(gamma_cp)
        
    return gamma_cp

##########
def FFTConvolve(in1, in2):
    
    #in1[in1==cp.nan]=0
    #in2[in2==cp.nan]=0
    #ret = cp.fft.ifft2(cp.fft.fft2(in1) * cp.fft.fft2(in2))
    
    return sp.fft.ifft2(sp.fft.fft2(in1) * sp.fft.fft2(in2))

#############################################################
#    FILTER FOR OSS
#############################################################
def W(npx,npy,alpha=0.1):
    '''
    Simple generator of a gaussian, used for filtering in OSS
    INPUT:  npx,npy: number of pixels on the image
            alpha: width of the gaussian 
            
    OUTPUT: gaussian matrix
    
    --------
    author: RB 2020
    '''
    Y,X = cp.meshgrid(cp.arange(npy),cp.arange(npx))
    k=(cp.sqrt((X-npx//2)**2+(Y-npy//2)**2))
    return cp.fft.fftshift(cp.exp(-0.5*(k/alpha)**2))

#############################################################
#    ERROR FUNCTIONS
#############################################################
def Error_diffract(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data. 
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns 
            
    OUTPUT: Error between the two
    
    --------
    author: RB 2020
    '''
    Num=(diffract-guess)**2
    Den=diffract**2
    Error = Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

def Error_diffract_cp(guess, diffract):
    '''
    Error on the diffraction attern of retrieved data. 
    INPUT:  guess, diffract: retrieved and experimental diffraction patterns 
            
    OUTPUT: Error between the two
    
    --------
    author: RB 2020
    '''
    Num=np.abs(diffract-guess)**2
    Den=np.abs(diffract)**2
    Error = Num.sum()/Den.sum()#cp.sum(Num)/cp.sum(Den)
    Error=10*np.log10(Error)
    return Error

def Error_support(prev,mask):
    '''
    Error on the support of retrieved data. 
    INPUT:  prev: retrieved image
            mask: support mask
            
    OUTPUT: Error on the support, how much prev is outside of "mask"
    
    --------
    author: RB 2020
    '''
    Num=prev*(1-mask)**2
    Den=prev**2
    Error=Num.sum()/Den.sum()
    Error=10*np.log10(Error)
    return Error

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

    if type(center)==type(None):
        center=np.array([CCD.shape[1]/2, CCD.shape[0]/2])


    print("center=",center, "z=",z )
    values=CCD.flatten()
    points=(np.array(np.unravel_index(np.arange(values.size), CCD.shape))).astype('float64')
    
    points[0,:]-=center[0]
    points[1,:]-=center[1]
    points*= experimental_setup['px_size']
    
    
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
    

    return CCD_projected

############################################
## Fourier Ring Correlation
############################################

def FRC0(im1,im2,width_bin):
    '''
    implements Fourier Ring Correlation. (https://www.nature.com/articles/s41467-019-11024-z)
    Input:  im1,im2: two diffraction patterns with different sources of noise. Can also use same image twice, sampling only odd/even pixels
            width_bin: width of disks we will use to have our histogram
            
    Output: sum_num: array of all numerators value of correlation hystogram
            sum_den: array of all denominators value of correlation hystogram
    
    -------
    author: RB 2020
    '''
    shape=im1.shape
    Num_bins=shape[0]//(2*width_bin)
    sum_num=np.zeros(Num_bins)
    sum_den=np.zeros(Num_bins)
    center = np.array([shape[0]//2, shape[1]//2])
    
    for i in range(Num_bins):
        annulus = np.zeros(shape)
        yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[1], center[0]), i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0

        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        sum_num[i]=np.sum( im1* np.conj(im2) * annulus )#np.sum( im1[np.nonzero(annulus)] * np.conj(im2[np.nonzero(annulus)]) )
        sum_den[i]=np.sqrt( np.sum(np.abs(im1)**2* annulus) * np.sum(np.abs(im2)**2* annulus) )
        
    return sum_num,sum_den

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def FRC(im1,im2,width_bin, center=0):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)'''
    
    shape=im1.shape
    Num_bins=shape[0]//(2*width_bin)
    sum_num=np.zeros(Num_bins)
    sum_den=np.zeros(Num_bins)
    if type(center)==int:
        center = np.array([shape[0]//2, shape[1]//2])
    
    FT1=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im1)))
    FT2=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im2)))
    
    for i in range(Num_bins):
        yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[1], center[0]), i*width_bin)
        
        yy,xx=np.where()
        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        sum_num[i]=np.sum(FT1* np.conj(FT2)[yy_outer, xx_outer]) - np.sum(FT1* np.conj(FT2)[yy_inner, xx_inner])
        sum_den[i]=np.sqrt( (np.sum(np.abs(FT1[yy_outer, xx_outer])**2)-np.sum(np.abs(FT1[yy_inner, xx_inner])**2)) * np.sum(np.abs(FT2[yy_outer, xx_outer])**2- np.abs(FT2[yy_inner, xx_inner])**2) )
        
    return sum_num,sum_den

def FRC_1image(im1,width_bin, output='average'):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 1 image in real space
            width of the bin, integer
            string to decide the output (optional)
    output: FRC istogram average, or array containing separate hystograms 01even-even-odd-odd, 23even-odd-odd-even, 20even-odd-even-even, 13odd-odd-odd-even'''
    shape=im1.shape
    Num_bins=shape[0]//(2*2*width_bin)
    sum_num=np.zeros((4,Num_bins))
    sum_den=np.zeros((4,Num_bins))
    
    #eveneven, oddodd, evenodd, oddeven
    im=[im1[::2, ::2],im1[1::2, 1::2],im1[::2, 1::2],im1[1::2, ::2]]
    FT1st=[0,2,2,1]
    FT2nd=[1,3,0,3]
    
    for j in range(0,4):
        
        sum_num[j,:],sum_den[j,:]=FRC(im[FT1st[j]],im[FT2nd[j]],width_bin)

    FRC_array=sum_num/sum_den        
    FRC_data=np.sum(FRC_array,axis=0)/4
    
    if output=='average':
        return FRC_data
    else:
        return FRC_array
    

    
def FRC_GPU2(im1,im2,width_bin,center=0, start_Fourier=True):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 2 images in real space
            width of the bin, integer
    output: FRC histogram array
    
    RB 2020'''
    

    
    shape=im1.shape
    shape=(int(np.ceil(np.sqrt(2)*shape[0])),int(np.ceil(np.sqrt(2)*shape[0])))
    Num_bins=shape[0]//(2*width_bin)
    
    sum_num=cp.zeros(Num_bins)
    sum_den=cp.zeros(Num_bins)
    
    if type(center)==int:
        center = np.array([im1.shape[0]//2, im1.shape[1]//2])
    
    if start_Fourier:
        FT1=cp.asarray(im1)
        FT2=cp.asarray(im2)
    else:
        im1_cp=cp.asarray(im1)
        im2_cp=cp.asarray(im2)
        FT1=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im1_cp)))
        FT2=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im2_cp)))

    for i in range(Num_bins):
        yy_outer, xx_outer = disk((center[0], center[1]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[0], center[1]), i*width_bin)
        if i==0:
            yy_inner, xx_inner = disk((center[0], center[1]), 1)

        outer=np.zeros((yy_outer.shape[0],2))
        outer[:,0]=xx_outer.copy()
        outer[:,1]=yy_outer.copy()

        in_delete=np.amax(outer, axis=1)>=(im1.shape[0])
        outer=np.delete(outer, in_delete, axis=0)
        in_delete=np.amin(outer, axis=1)<(0)
        outer=np.delete(outer, in_delete, axis=0)

        inner=np.zeros((yy_inner.shape[0],2))
        inner[:,0]=xx_inner.copy()
        inner[:,1]=yy_inner.copy()

        in_delete=np.amax(inner, axis=1)>=(im1.shape[0])
        inner=np.delete(inner, in_delete, axis=0)
        in_delete=np.amin(inner, axis=1)<(0)
        inner=np.delete(inner, in_delete, axis=0)

        inner=np.rint(inner).astype(int)
        outer=np.rint(outer).astype(int)
        #print(".",inner.size,outer.size)
        sum_num[i]=cp.sum( (FT1* cp.conj(FT2))[outer[:,1], outer[:,0]] ) - cp.sum( (FT1* cp.conj(FT2))[inner[:,1], inner[:,0]] )
        sum_den[i]=cp.sqrt( (cp.sum(cp.abs(FT1[outer[:,1], outer[:,0]])**2)-cp.sum(cp.abs(FT1[inner[:,1], inner[:,0]])**2)) * (cp.sum(cp.abs(FT2[outer[:,1], outer[:,0]])**2)-cp.sum(cp.abs(FT2[inner[:,1], inner[:,0]])**2)) )
 

        #sum_num[i]=cp.sum( (FT1* cp.conj(FT2))[yy_outer, xx_outer] ) - cp.sum( (FT1* cp.conj(FT2))[yy_inner, xx_inner] )
        #sum_den[i]=cp.sqrt( (cp.sum(cp.abs(FT1[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT1[yy_inner, xx_inner])**2)) * (cp.sum(cp.abs(FT2[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT2[yy_inner, xx_inner])**2)) )
        
    FRC_array=sum_num/sum_den
    FRC_array_np=cp.asnumpy(FRC_array)
    
    return FRC_array_np
    
def FRC_GPU(im1,im2,width_bin,center=0, start_Fourier=True):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 2 images in real space
            width of the bin, integer
    output: FRC histogram array
    
    RB 2020'''
    

    
    shape=im1.shape
    #shape=(int(np.ceil(np.sqrt(2)*shape[0])),int(np.ceil(np.sqrt(2)*shape[0])))
    Num_bins=shape[0]//(2*width_bin)
    
    sum_num=cp.zeros(Num_bins)
    sum_den=cp.zeros(Num_bins)
    
    if type(center)==int:
        center = np.array([shape[0]//2, shape[1]//2])
    
    if start_Fourier:
        FT1=cp.asarray(im1)
        FT2=cp.asarray(im2)
    else:
        im1_cp=cp.asarray(im1)
        im2_cp=cp.asarray(im2)
        FT1=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im1_cp)))
        FT2=cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(im2_cp)))

    for i in range(Num_bins):
        yy_outer, xx_outer = disk((center[0], center[1]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[0], center[1]), i*width_bin)
        sum_num[i]=cp.sum( (FT1* cp.conj(FT2))[yy_outer, xx_outer] ) - cp.sum( (FT1* cp.conj(FT2))[yy_inner, xx_inner] )
        sum_den[i]=cp.sqrt( (cp.sum(cp.abs(FT1[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT1[yy_inner, xx_inner])**2)) * (cp.sum(cp.abs(FT2[yy_outer, xx_outer])**2)-cp.sum(cp.abs(FT2[yy_inner, xx_inner])**2)) )
        
    FRC_array=sum_num/sum_den
    FRC_array_np=cp.asnumpy(FRC_array)
    
    return FRC_array_np


def FRC_1image_GPU(im1,width_bin, output='average'):
    '''implements Fourier Ring Correlation. 
    RB June 2020 (https://www.nature.com/articles/s41467-019-11024-z)
    
    INPUT: 1 image in real space
            width of the bin, integer
            string to decide the output (optional)
    output: FRC istogram average, or array containing separate hystograms 01even-even-odd-odd, 23even-odd-odd-even, 20even-odd-even-even, 13odd-odd-odd-even
    
    RB 2020'''
    
    shape=im1.shape
    Num_bins=shape[0]//(2*2*width_bin)
    FRC_array=np.zeros((4,Num_bins))
    
    #eveneven, oddodd, evenodd, oddeven
    im=[im1[::2, ::2],im1[1::2, 1::2],im1[::2, 1::2],im1[1::2, ::2]]
    FT1st=[0,2,2,1]
    FT2nd=[1,3,0,3]
    
    for j in range(0,4):
        
        FRC_array[j,:]=FRC_GPU(im[FT1st[j]],im[FT2nd[j]],width_bin)
      
    FRC_data=np.sum(FRC_array,axis=0)/4
    
    if output=='average':
        return FRC_data
    else:
        return FRC_array
    
def half_bit_thrs(shape, SNR=0.5, width_bin=5, center=0):
    '''van heel and schatz 2005
    gives you an array containing values for the half bit threshold
    RB 2020'''

    Num_bins=(np.ceil(np.sqrt(2)*shape[0])//(2*width_bin)).astype(int)
    
    if type(center)==int:
        center = np.array([shape[0]//2, shape[1]//2])
    
    thr=np.zeros(Num_bins)
    
    for i in range(Num_bins):
        yy_outer, xx_outer = disk((center[0], center[1]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[0], center[1]), i*width_bin)
        if i==0:
            yy_inner, xx_inner = disk((center[0], center[1]), 1)

        outer=np.zeros((yy_outer.shape[0],2))
        outer[:,0]=xx_outer.copy()
        outer[:,1]=yy_outer.copy()

        in_delete=np.amax(outer, axis=1)>=(shape[0])
        outer=np.delete(outer, in_delete, axis=0)
        in_delete=np.amin(outer, axis=1)<(0)
        outer=np.delete(outer, in_delete, axis=0)

        inner=np.zeros((yy_inner.shape[0],2))
        inner[:,0]=xx_inner.copy()
        inner[:,1]=yy_inner.copy()

        in_delete=np.amax(inner, axis=1)>=(shape[0])
        inner=np.delete(inner, in_delete, axis=0)
        in_delete=np.amin(inner, axis=1)<(0)
        inner=np.delete(inner, in_delete, axis=0)

        inner=np.rint(inner).astype(int)
        outer=np.rint(outer).astype(int)

        n=outer.size-inner.size
        #print(outer.size,inner.size,n)
    
        thr[i]=(SNR+ (2*np.sqrt(SNR)+1)/np.sqrt(n))/(SNR+1+2*np.sqrt(SNR)/np.sqrt(n))
    return thr

def PRTF(im, exp_im, width_bin=5):
    '''function for Phase Retrieval Transfer Function
    RB Jan 2021
    INPUT: im: sums of retrieved image
            exp_im: experimental scattering pattern
            width_bin: width of bins used to plot PRTF
    output: prtf: phase retrieval transfer function'''
    
    prtf= im/exp_im
    
    prtf_cp=cp.asarray(prtf)
    
    shape=prtf.shape
    Num_bins=shape[0]//(2*width_bin)
    
    prtf_array=cp.zeros(Num_bins)
    center = np.array([shape[0]//2, shape[1]//2])

    
    for i in range(Num_bins):
        annulus = cp.zeros(shape)
        yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
        yy_inner, xx_inner = disk((center[1], center[0]), i*width_bin)
        annulus[yy_outer,xx_outer]=1
        annulus[yy_inner,xx_inner]=0

        #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
        prtf_array[i]=cp.sum( prtf_cp * annulus )/cp.sum(annulus)
        
    prtf_array_np=cp.asnumpy(prtf_array)
    
    return prtf_array_np


def azimutal_integral_GPU_3(im,mask=1,width_bin=1):
    '''azimuthal integral of an image, or of an arry of images. It understands alone if it's a list of images or a numpy array of three dimensions
    Input: im: image / array of images / list of images to be done hazimuthal average of
            mask: image  to mask some defective pixels. =0 for pixels to be masked. FOR NOW IT DOES NOT WORK
           width bin: width of the bin to be considered
    Output: array/array of arrays/list of arrays representing the azimuthal integral
    RB 2021'''

    if type(mask) is int:
        mask_cp=mask
        
    elif type(mask) is np.ndarray and mask.ndim==2:
        mask_cp=cp.asarray(mask) 
    
    
    if type(im) is np.ndarray and im.ndim==2:
        shape=im.shape 
        center = cp.array([shape[0]//2, shape[1]//2])
        y, x = cp.indices((im.shape))
        r = cp.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(cp.int)
        r[mask_cp==0]=shape[0]*3 # just to exclude them
        im_cp=cp.asarray(im)

         
        Num_bins=int(((np.sqrt(2)*shape[0])//(2*width_bin)))
        azimuthal_integral=cp.zeros(Num_bins)
                
        for i in cp.arange(Num_bins):
            azimuthal_integral[i]=cp.average(im_cp[(r>=i*width_bin) & (r<(i+1)*width_bin)])

            
    else:
        if type(im) is list: #so im is a numpy array of dimension 3
            Num_images=len(im)
            im_cp=[0]*Num_images
            
            for i in range(Num_images):
                im_cp[i]=cp.asarray(im[i])
            shape=im[0].shape  
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=[cp.zeros(Num_bins)]*Num_images

        else: #list or 2D np array
            im_cp=cp.asarray(im)
            Num_images=im.shape[0]
            shape=im[0].shape
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=cp.zeros((Num_images,Num_bins))
        
        center = np.array([shape[0]//2, shape[1]//2])

        for i in range(Num_bins):
            yy_outer, xx_outer = disk((center[0], center[1]), (i+1)*width_bin)
            yy_inner, xx_inner = disk((center[0], center[1]), i*width_bin)
            if i==0:
                yy_inner, xx_inner = disk((center[0], center[1]), 1)

            outer=np.zeros((yy_outer.shape[0],2))
            outer[:,0]=xx_outer.copy()
            outer[:,1]=yy_outer.copy()

            in_delete=np.amax(outer, axis=1)>=(im1.shape[0])
            outer=np.delete(outer, in_delete, axis=0)
            in_delete=np.amin(outer, axis=1)<(0)
            outer=np.delete(outer, in_delete, axis=0)

            inner=np.zeros((yy_inner.shape[0],2))
            inner[:,0]=xx_inner.copy()
            inner[:,1]=yy_inner.copy()

            in_delete=np.amax(inner, axis=1)>=(im1.shape[0])
            inner=np.delete(inner, in_delete, axis=0)
            in_delete=np.amin(inner, axis=1)<(0)
            inner=np.delete(inner, in_delete, axis=0)

            inner=np.rint(inner).astype(int)
            outer=np.rint(outer).astype(int)
        
            
            for j in range(Num_images):
                azimuthal_integral[j][i]=(cp.sum( im_cp[j][outer[:,1],outer[:,0]]) -cp.sum(im_cp[j][inner[:,1],inner[:,0]])) / (outer[:,1].size-inner[:,1].size)
                
    azimuthal_integral_np=cp.asnumpy(azimuthal_integral)
    
    return azimuthal_integral_np



def azimutal_integral_GPU_2(im,mask=1,width_bin=1):
    '''azimuthal integral of an image, or of an arry of images. It understands alone if it's a list of images or a numpy array of three dimensions
    Input: im: image / array of images / list of images to be done hazimuthal average of
            mask: image  to mask some defective pixels. =0 for pixels to be masked. FOR NOW IT DOES NOT WORK
           width bin: width of the bin to be considered
    Output: array/array of arrays/list of arrays representing the azimuthal integral
    RB 2021'''

    if type(mask) is int:
        mask_cp=mask
        
    elif type(mask) is np.ndarray and mask.ndim==2:
        mask_cp=cp.asarray(mask) 
    
    
    if type(im) is np.ndarray and im.ndim==2:
        shape=im.shape 
        center = cp.array([shape[0]//2, shape[1]//2])
        y, x = cp.indices((im.shape))
        r = cp.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(cp.int)
        r[mask_cp==0]=shape[0]*3 # just to exclude them
        im_cp=cp.asarray(im)

         
        Num_bins=int(((np.sqrt(2)*shape[0])//(2*width_bin)))
        azimuthal_integral=cp.zeros(Num_bins)
                
        for i in cp.arange(Num_bins):
            azimuthal_integral[i]=cp.average(im_cp[(r>=i*width_bin) & (r<(i+1)*width_bin)])

            
    else:
        if type(im) is list: #so im is a numpy array of dimension 3
            Num_images=len(im)
            im_cp=[0]*Num_images
            
            for i in range(Num_images):
                im_cp[i]=cp.asarray(im[i])
            shape=im[0].shape  
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=[cp.zeros(Num_bins)]*Num_images

        else: #list or 2D np array
            im_cp=cp.asarray(im)
            Num_images=im.shape[0]
            shape=im[0].shape
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=cp.zeros((Num_images,Num_bins))
        
        center = np.array([shape[0]//2, shape[1]//2])

        
        yy_inner, xx_inner = disk((center[1], center[0]), 0)
        for i in range(Num_bins):
            yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
            
            for j in range(Num_images):
                azimuthal_integral[j][i]=(cp.sum( im_cp[j][yy_outer,xx_outer]) -cp.sum(im_cp[j][yy_inner,xx_inner])) / (yy_outer.size-yy_inner.size)
                
            yy_inner,xx_inner=yy_outer.copy(),xx_outer.copy()
                
    azimuthal_integral_np=cp.asnumpy(azimuthal_integral)
    
    return azimuthal_integral_np

def azimutal_integral_GPU(im,mask=1,width_bin=1):
    '''azimuthal integral of an image, or of an arry of images. It understands alone if it's a list of images or a numpy array of three dimensions
    Input: im: image / array of images / list of images to be done hazimuthal average of
            mask: image  to mask some defective pixels. =0 for pixels to be masked. FOR NOW IT DOES NOT WORK
           width bin: width of the bin to be considered
    Output: array/array of arrays/list of arrays representing the azimuthal integral
    RB 2021'''
    
    if type(mask) is int:
        mask_cp=mask
        
    elif type(mask) is np.ndarray and mask.ndim==2:
        mask_cp=cp.asarray(mask) 
    
    
    if type(im) is np.ndarray and im.ndim==2:
        print("array 2D")
        im_cp=cp.asarray(im)
        shape=im.shape  
        Num_bins=shape[0]//(2*width_bin)
        azimuthal_integral=cp.zeros(Num_bins)
        
        center = np.array([shape[0]//2, shape[1]//2])
        #annulus = cp.zeros(shape)
        
        yy_inner, xx_inner = disk((center[1], center[0]), 0)
        
        for i in range(Num_bins):
            yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
            #annulus[yy_outer,xx_outer]=1
            #annulus[yy_inner,xx_inner]=0
            #annulus*=mask_cp

            #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
            #azimuthal_integral[i]=cp.sum( im_cp * annulus ) / cp.sum( annulus )
            azimuthal_integral[i]=(cp.sum( im_cp[yy_outer,xx_outer])- cp.sum(im_cp[yy_inner,xx_inner])) / (yy_outer.size-yy_inner.size)
            yy_inner,xx_inner=yy_outer.copy(),xx_outer.copy()
            
    else:
        if type(im) is list: #so im is a numpy array of dimension 3
            print("list")
            Num_images=len(im)
            im_cp=[0]*Num_images
            
            for i in range(Num_images):
                im_cp[i]=cp.asarray(im[i])
            shape=im[0].shape  
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=[cp.zeros(Num_bins)]*Num_images

            

        else: #list or 2D np array
            print("array 3D")
            im_cp=cp.asarray(im)
            Num_images=im.shape[0]
            shape=im[0].shape
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=cp.zeros((Num_images,Num_bins))
        
        center = np.array([shape[0]//2, shape[1]//2])
        #annulus = cp.zeros(shape)
        
        yy_inner, xx_inner = disk((center[1], center[0]), 0)
        for i in range(Num_bins):
            yy_outer, xx_outer = disk((center[1], center[0]), (i+1)*width_bin)
            #annulus[yy_outer,xx_outer]=1
            #annulus[yy_inner,xx_inner]=0
            
            if i%100==0:
                print("bin=",i)
            for j in range(Num_images):
                #a for cycle going through the various rings, summing up the terms for the denominator and calculating each term in the ring
                #azimuthal_integral[j][i]=cp.sum( im_cp[j] * annulus*mask_cp ) / cp.sum( annulus*mask_cp )
                azimuthal_integral[j][i]=(cp.sum( im_cp[j][yy_outer,xx_outer]) -cp.sum(im_cp[j][yy_inner,xx_inner])) / (yy_outer.size-yy_inner.size)
                
            yy_inner,xx_inner=yy_outer.copy(),xx_outer.copy()
                
    azimuthal_integral_np=cp.asnumpy(azimuthal_integral)
    
    return azimuthal_integral_np

def azimutal_integral(im,mask=1,width_bin=1):
    '''azimuthal integral of an image, or of an arry of images. It understands alone if it's a list of images or a numpy array of three dimensions
    Input: im: image / array of images / list of images to be done hazimuthal average of
            mask: image  to mask some defective pixels. =0 for pixels to be masked
           width bin: width of the bin to be considered
    Output: array/array of arrays/list of arrays representing the azimuthal integral
    RB 2021'''

    
    
    if type(im) is np.ndarray and im.ndim==2:
        print("array 2D")

        shape=im.shape  
        Num_bins=shape[0]//(2*width_bin)
        
        center = np.array([shape[0]//2, shape[1]//2])
        
        azimuthal_integral=radial_profile(im, center)
        
            
    else:
        if type(im) is list: #so im is a numpy array of dimension 3
            print("list")
            Num_images=len(im)
            im_cp=[0]*Num_images
            
            for i in range(Num_images):
                im_cp[i]=cp.asarray(im[i])
                
            shape=im[0].shape  
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=[np.zeros(Num_bins)]*Num_images

            

        else: #list or 2D np array
            print("array 3D")
            Num_images=im.shape[0]
            shape=im[0].shape
            Num_bins=shape[0]//(2*width_bin)
            azimuthal_integral=np.zeros((Num_images,Num_bins))
        
        center = np.array([shape[0]//2, shape[1]//2])

        for j in range(Num_images):
            azimuthal_integral[j]=radial_profile(im[j], center)
                
    azimuthal_integral_np=cp.asnumpy(azimuthal_integral)
    
    return azimuthal_integral_np



def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile





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
    
    
def spectrogram(pos, neg, N=10, mode="+", figsize=(10,10)):
    
    image=np.zeros(pos.shape, dtype=np.complex)
    width=pos.shape[0]//N
    
    if mode=="+":
        fth_image=pos+neg
    elif mode=="-":
        fth_image=pos-neg
        
    for i in range(N):
        for j in range(N):
            image[ i* width: (i+1)*width, j* width: (j+1)*width] = fth.reconstruct(fth_image[ i* width: (i+1)*width, j* width: (j+1)*width])
            
    mi,ma=np.percentile(np.abs(image), (1,99))
    fig,ax=plt.subplots(figsize=figsize)
    ax.imshow(np.abs(image), vmin=mi, vmax=ma)
    
    return image
