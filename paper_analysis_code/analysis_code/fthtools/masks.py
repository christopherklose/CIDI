import numpy as np
from scipy.ndimage import gaussian_filter


def circle(radius):
    """Returns array of shape (size, size) with circular region set to one."""
    return np.hypot(*np.ogrid[-radius:radius, -radius:radius]) <= radius


def add_circle(image, c0, c1, r, value=1):
    """Add circular ragion to image (in-place!)"""
    image[c0 - r:c0 + r, c1 - r:c1 + r] = value * circle(r)
    return image


def mask_circle(shape, radius, sigma=1, center=None):
    '''
    Returns a mask with unity value and a circular region set to zero.
    
    Parameters
    ----------
    shape: length-2 sequence
        Shape of the mask to generate
    radius: float
        circle radius
    sigma: float, optional
        Optional smoothing of the mask (sigma=0 disables smoothing).
        See scipy.ndimage.gaussian_filter for details.
    center: sequence of floats, optional
        Position of the circular region. Defaults to image center.
    
    Returns
    -------
    mask: array
        Masking array
    '''
    n0, n1 = shape
        
    if center is None:
        c0, c1 = n0 / 2, n1 / 2
    else:
        c0, c1 = center

    xx, yy = np.ogrid[:n0, :n1]
    xx = xx - c0
    yy = yy - c1
    mask = np.hypot(xx, yy)
    mask = (mask > radius).astype(float)
    if sigma > 0:
        mask = gaussian_filter(mask, sigma, mode="constant", cval=1)
    return mask