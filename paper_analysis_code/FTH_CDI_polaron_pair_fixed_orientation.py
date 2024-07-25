## import packages
import numpy as np
import scipy as sp
import pyqtgraph_extended as pg
import matplotlib.pyplot as plt
from matplotlib import pyplot
import holography
import array as array
from scipy import signal
from numpy import genfromtxt

import sys
sys.path.append(r"folder_for_MPI_Berlin\analysis_code")
import fthtools.masks as masks
import fthtools.fth as fth
import fthtools.PhaseRetrieval as PhR
import matplotlib.image as mpimg
import os


##Function to define stochastic noise
def gauss_2D(xx,yy,amp,sigma, x0,y0):
    return amp*np.exp(-((xx-x0)**2+(yy-y0)**2)/(sigma)**2)

##Function to convert rgb image to gray scale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

## import image
img = mpimg.imread(r'folder_for_MPI_Berlin\sample_image.png')
image1 = rgb2gray(img)


## Define clip matrix
pixeln=int(np.sqrt(image1.size))
clip_radius=40
rows, cols = pixeln,pixeln
row_vec = np.double(np.arange(0,rows))
col_vec = np.double(np.arange(0,cols))
yy, xx= np.meshgrid(rows//2-col_vec, cols//2-row_vec)
mask =1-((xx)**2 + (yy)**2>clip_radius**2)
mask1= mask

Final_error_list=[]

## define holography hole

r_hole_radius1=2
r_hole_radius2=2
location_r1x= 180
location_r1y= 150
location_r2x= 100
location_r2y= -150
holography_hole1=(1-((xx+location_r1x)**2 + (yy+location_r1y)**2>r_hole_radius1**2))*1
holography_hole2=(1-((xx+location_r2x)**2 + (yy+location_r2y)**2>r_hole_radius2**2))*1
holography_hole=holography_hole1+holography_hole2


##number of diffraction pattern available
n=500

## Initialization of matrices
incoherent_CDI=np.zeros_like(mask)
av_pattern=np.zeros_like(mask)
ER2s=np.zeros_like(mask, dtype=complex)
EsR1=np.zeros_like(mask, dtype=complex)
ER1s=np.zeros_like(mask, dtype=complex)
EsR2=np.zeros_like(mask, dtype=complex)
R1R2s=np.zeros_like(mask, dtype=complex)
R1sR2=np.zeros_like(mask, dtype=complex)
Sktav=np.zeros_like(mask)
F_Sktav=np.zeros_like(mask)
FER2s=np.zeros_like(mask)
FEsR1=np.zeros_like(mask)
FER1s=np.zeros_like(mask)
FEsR2=np.zeros_like(mask)
FR1R2s=np.zeros_like(mask)
FR1sR2=np.zeros_like(mask)
F_Sktav=np.zeros_like(mask)
F_fluctuation=np.zeros_like(mask)
F_fluctuation_mean=np.zeros_like(mask)
F_fluctuation_sq_mean=np.zeros_like(mask)
FAutoCorl1=np.zeros_like(mask)
FAutoCorl=np.zeros_like(mask)
FAutoCorlN=np.zeros_like(mask)
FAutoCorlN2=np.zeros_like(mask)
Av_fluctuation=np.zeros_like(mask)
AvSq_fluctuation=np.zeros_like(mask)

mask_scatter = np.ones_like(mask) #for CDI part

##Add holography hole
image1=image1*mask+holography_hole

##Define phase
phase1= (100*(xx+yy)/(pixeln*np.pi))*mask1

##Averaging dataset
for ii in np.arange(n):
    print(ii)
    fluctuation=np.zeros_like(mask)
    x_f=np.random.random()*60-30
    y_f=np.random.random()*60-30
    fluctuation=gauss_2D(xx,yy,1,2,x_f,y_f) -gauss_2D(xx,yy,1,2,x_f+40,y_f)
    fluctuation=fluctuation*mask
    pattern=1*fluctuation+1*image1
    diffraction=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(pattern)))
    incoherent_CDI=incoherent_CDI+np.abs(diffraction)**2
    F_fluctuation=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift((fluctuation)*mask)))
    F_fluctuation_mean=F_fluctuation_mean+(F_fluctuation)
    F_fluctuation_sq_mean=F_fluctuation_sq_mean+np.abs(F_fluctuation)**2
    Av_fluctuation=Av_fluctuation+fluctuation
    av_pattern= av_pattern+ pattern
    # pg.image(np.abs(fluctuation)+mask)

##Fourier transform of averaged hologram
hologram2=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(incoherent_CDI/n)))

##Extract auto-correlations and cross correlations separately
RadER1= clip_radius+r_hole_radius1+10
RadER2= clip_radius+r_hole_radius2+10
RadRR= r_hole_radius1+r_hole_radius2+10
##E*R1
EsR1[((rows//2+location_r1x)-RadER1):((rows//2+location_r1x)+RadER1),((cols//2+location_r1y)-RadER1):((cols//2+location_r1y)+RadER1)]=hologram2[((rows//2+location_r1x)-RadER1):((rows//2+location_r1x)+RadER1),((cols//2+location_r1y)-RadER1):((cols//2+location_r1y)+RadER1)]
FEsR1=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift( EsR1)))
##ER1*
ER1s[((rows//2-location_r1x)-RadER1):((rows//2-location_r1x)+RadER1),((cols//2-location_r1y)-RadER1):((cols//2-location_r1y)+RadER1)]=hologram2[((rows//2-location_r1x)-RadER1):((rows//2-location_r1x)+RadER1),((cols//2-location_r1y)-RadER1):((cols//2-location_r1y)+RadER1)]
FER1s=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ER1s)))
##E*R2
EsR2[((rows//2+location_r2x)-RadER2):((rows//2+location_r2x)+RadER2),((cols//2+location_r2y)-RadER2):((cols//2+location_r2y)+RadER2)]=hologram2[((rows//2+location_r2x)-RadER2):((rows//2+location_r2x)+RadER2),((cols//2+location_r2y)-RadER2):((cols//2+location_r2y)+RadER2)]
FEsR2=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(EsR2)))
##ER2*
ER2s[((rows//2-location_r2x)-RadER2):((rows//2-location_r2x)+RadER2),((cols//2-location_r2y)-RadER2):((cols//2-location_r2y)+RadER2),]=hologram2[((rows//2-location_r2x)-RadER2):((rows//2-location_r2x)+RadER2),((cols//2-location_r2y)-RadER2):((cols//2-location_r2y)+RadER2)]
FER2s=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift( ER2s)))

##R1R2*
R1R2s[rows//2+(location_r1x-location_r2x)-RadRR:rows//2+(location_r1x-location_r2x)+RadRR, cols//2+(location_r1y-location_r2y)-RadRR:cols//2+(location_r1y-location_r2y)+RadRR]=hologram2[rows//2+(location_r1x-location_r2x)-RadRR:rows//2+(location_r1x-location_r2x)+RadRR, cols//2+(location_r1y-location_r2y)-RadRR:cols//2+(location_r1y-location_r2y)+RadRR]
FR1R2s=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(R1R2s)))

##R1*R2
R1sR2[rows//2-(location_r1x-location_r2x)-RadRR:rows//2-(location_r1x-location_r2x)+RadRR, cols//2-(location_r1y-location_r2y)-RadRR:cols//2-(location_r1y-location_r2y)+RadRR]=hologram2[rows//2-(location_r1x-location_r2x)-RadRR:rows//2-(location_r1x-location_r2x)+RadRR, cols//2-(location_r1y-location_r2y)-RadRR:cols//2-(location_r1y-location_r2y)+RadRR]
FR1sR2=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(R1sR2)))

##|E|^4
FAutoCorl=(FER2s*FEsR1*FER1s*FEsR2)/(FR1R2s*FR1sR2)

##|R1|^2
FR1R1s=np.abs((FER1s*FEsR1)/np.abs(FAutoCorl)**0.5)

##|R2|^2
FR2R2s=np.abs((FER2s*FEsR2)/np.abs(FAutoCorl)**0.5)

##Stochastic term from CIDI
F_Sktav=incoherent_CDI/n-(np.abs(FAutoCorl)**0.5+FR1R1s+FR2R2s)-(FR1R2s+FR1sR2+FER2s+FEsR1+FER1s+FEsR2)

##CDI part

##Image of the isolated stochastic term in object plane (Used to define proper mask to perform CDI)
Sktav=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(F_Sktav)))
support=(np.abs(Sktav) > 2e-6).astype(int)


## Square root of diffrction pattern for CDI
SQRT_F_Sktav=np.abs(F_Sktav**0.5)


##Adding mask to numerical zero errors
mask_scatter=1-((xx)**2 + (yy)**2>230**2)+((xx)**2 + (yy)**2>240**2)
SQRT_F_Sktav=(SQRT_F_Sktav*mask_scatter)


initial_guess= np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask)))

x = np.clip(SQRT_F_Sktav, 0, None)
y = np.clip(np.abs(initial_guess), 0, None)
res = sp.stats.linregress(x.flatten(), y.flatten())
initial_guess -= res.intercept
initial_guess /= res.slope

initial_guess = (SQRT_F_Sktav  * np.exp(1j * np.angle(initial_guess)))


SW_freq = 1e4  # disable
Nit = 500

##
retrieved_res0, Error_diff_p, Error_supp, supportmask = PhR.PhaseRtrv_CPU(
    diffract=SQRT_F_Sktav,
    mask=support*mask,
    mode="mine",
    beta_zero=0.5,
    Nit=Nit,
    beta_mode='arctan',
    plot_every=20,
    # Phase=initial_guess,
    Phase=0,
    seed=False,
    real_object=False,
    bsmask=(1 - mask_scatter),
    average_img=20,
    Fourier_last=True,
    SW_freq=SW_freq
)

ret_pattern = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(retrieved_res0)))

plt.clf()
plot1=plt.pcolormesh(row_vec, col_vec, np.abs(ret_pattern)/np.max(np.abs(ret_pattern)), rasterized = True, cmap='Blues')
plt.colorbar(plot1)
plt.xlim(360, 380)
plt.ylim(325, 380)
plt.clim(0,1)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()