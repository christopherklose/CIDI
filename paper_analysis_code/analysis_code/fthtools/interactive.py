import numpy as np
import h5py

import scipy as sp
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
from ipywidgets import FloatRangeSlider, FloatSlider, Button, interact, IntSlider
from scipy.constants import c, h, e

import matplotlib.pyplot as plt
import ipywidgets

from .fth import reconstruct, shift_image, propagate, shift_phase


def cimshow(im, **kwargs):
    """Simple 2d image plot with adjustable contrast.
    
    Returns matplotlib figure and axis created.
    """
    im = np.array(im)
    fig, ax = plt.subplots()
    im0 = im[0] if len(im.shape) == 3 else im
    mm = ax.imshow(im0, **kwargs)

    cmin, cmax, vmin, vmax = np.percentile(im, [.1, 99.9, .001, 99.999])
    # vmin, vmax = np.nanmin(im), np.nanmax(im)
    sl_contrast = FloatRangeSlider(
        value=(cmin, cmax), min=vmin, max=vmax, step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width='500px'),
    )

    @ipywidgets.interact(contrast=sl_contrast)
    def update(contrast):
        mm.set_clim(contrast)
    
    if len(im.shape) == 3:
        w_image = IntSlider(value=0, min=0, max=im.shape[0] - 1)
        @ipywidgets.interact(nr=w_image)
        def set_image(nr):
            mm.set_data(im[nr])
    
    
    return fig, ax


class InteractiveCenter:
    """Plot image with controls for contrast and beamstop alignment tools."""
    def __init__(self, im, c0=None, c1=None, rBS=15, **kwargs):
        im = np.array(im)
        self.fig, self.ax = cimshow(im, **kwargs)
        self.mm = self.ax.get_images()[0]
        
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        
        self.c0 = c0
        self.c1 = c1
        self.rBS = rBS
        
        self.circles = []
        for i in range(5):
            color = 'g' if i == 1 else 'r'
            circle = plt.Circle([c0, c1], 10 * (i + 1), ec=color, fill=False)
            self.circles.append(circle)
            self.ax.add_artist(circle)

        w_c0 = ipywidgets.IntText(value=c0, description="c0")
        w_c1 = ipywidgets.IntText(value=c1, description="c1")
        w_rBS = ipywidgets.IntText(value=rBS, description="rBS")
        
        ipywidgets.interact(self.update, c0=w_c0, c1=w_c1, r=w_rBS)
    
    def update(self, c0, c1, r):
        self.c0 = c0
        self.c1 = c1
        self.rBS = r
        for i, c in enumerate(self.circles):
            c.set_center([c1, c0])
            c.set_radius(r * (i + 1))

def axis_to_roi(axis):
    """
    Generate numpy slice expression from bounds of matplotlib figure axis.
    """
    x0, x1 = sorted(axis.get_xlim())
    y0, y1 = sorted(axis.get_ylim())
    return np.s_[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]



class InteractiveOptimizer:
    """
    Interactively adjust FTH parameters: center, propagation and phase shift.
    
    TODO: parameters...
    """
    
    params = {"phase": 0, "center": (0, 0), "propdist": 0, "pixelsize": 13.5e-6,
              "energy": 779, "detectordist": 0.2}
    widgets = {}
    
    def __init__(self, holo, roi, params={}):
        self.params.update(params)
        self.holo = holo  #.astype(np.single)
        self.holo_centered = holo.copy()
        self.holo_prop = holo.copy()
        self.roi = roi
        
        self.make_ui()
    
    def make_ui(self):
        self.fig, (self.axr, self.axi) = plt.subplots(
            ncols=2, figsize=(7, 3.5), sharex=True, sharey=True,
            constrained_layout=True,
        )
        
        self.reco = reconstruct(self.holo)[self.roi]
        vmin, vmax = np.percentile(self.reco.real, [.01, 99.9])
        vlim = 2 * np.abs(self.reco.real).max()

        opt = dict(vmin=vmin, vmax=vmax, cmap="gray_r")
        self.mm_real = self.axr.imshow(self.reco.real, **opt)
        self.mm_imag = self.axi.imshow(self.reco.imag, **opt)
    
        self.widgets["clim"] = FloatRangeSlider(
            value=(vmin, vmax), min=-vlim, max=vlim,
        )
        self.widgets["phase"] = FloatSlider(
            value=self.params["phase"], min=-np.pi, max=np.pi,
        )
        self.widgets["c0"] = FloatSlider(
            value=self.params["center"][0], min=-5, max=5, step=.01
        )
        self.widgets["c1"] = FloatSlider(
            value=self.params["center"][1], min=-5, max=5, step=.01
        )
        self.widgets["propdist"] = FloatSlider(
            value=self.params["propdist"], min=-10, max=10, step=.1
        )
        self.widgets["energy"] = ipywidgets.BoundedFloatText(
            value=self.params["energy"], min=1, max=10000,
        )
        self.widgets["detectordist"] = ipywidgets.BoundedFloatText(
            value=self.params["detectordist"], min=.01
        )
        self.widgets["pixelsize"] = ipywidgets.BoundedFloatText(
            value=self.params["pixelsize"], min=1e-7,
        )
        
        interact(self.update_clim, clim=self.widgets["clim"])
        interact(self.update_phase, phase=self.widgets["phase"])
        interact(
            self.update_center,
            c0=self.widgets["c0"],
            c1=self.widgets["c1"]
        )
        interact(
            self.update_propagation,
            dist=self.widgets["propdist"],
            det=self.widgets["detectordist"],
            pxs=self.widgets["pixelsize"],
            energy=self.widgets["energy"],
        )
    
    def update_clim(self, clim):
        self.mm_real.set_clim(clim)
        self.mm_imag.set_clim(clim)
    
    def update_phase(self, phase):
        self.params["phase"] = phase
        reco_shifted = shift_phase(self.reco, phase)
        self.mm_real.set_data(reco_shifted.real)
        self.mm_imag.set_data(reco_shifted.imag)
    
    def update_center(self, c0, c1):
        self.params["center"] = (c0, c1)
        self.holo_centered = shift_image(self.holo, [c0, c1])
        self.reco = reconstruct(self.holo_centered)[self.roi]
        self.update_phase(self.params["phase"])
    
    def update_propagation(self, dist, det, pxs, energy):
        dist *= 1e-6
        self.params.update({
            "propdist": dist,
            "detectordist": det,
            "pixelsize": pxs,
            "energy": energy
        })
        self.holo_prop = propagate(self.holo_centered, dist, det, pxs, energy)
        self.reco = reconstruct(self.holo_prop)[self.roi]
        self.update_phase(self.params["phase"])
    
    def get_full_reco(self):
        return shift_phase(reconstruct(self.holo_prop), self.params["phase"])


def intensity_scale(im1, im2, mask=None):
    mask = mask if mask is not None else 1
    diff = (im1 - im2) * mask
    fig, ax = plt.subplots()
    hist, bins, patches = ax.hist(mask.flatten(), np.linspace(-100, 100, 201))
    ax.set_yscale("log")
    ax.axvline(0, c='r', lw=.5)
    ax.grid(True)

    @ipywidgets.interact(f=(.2, 2.0, .001))
    def update(f):
        diff = mask * (im1 - f * im2)
        hist, _ = np.histogram(diff, bins)
        for p, v in zip(patches, hist):
            p.set_height(v)
    return fig, ax
    