# Adaptet from MATLAB phantom3d.m by Jeff Fessler https://de.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom

import numpy as np

def phantom3d(size_or_def=None, size_n=None):
    """
    Generates a 3D head phantom (Shepp-Logan variant) for 3D reconstruction testing.
    
    Parameters
    ----------
    size_or_def : str, int, or ndarray, optional
        - If string: Specifies phantom type ('shepp-logan', 'modified shepp-logan', 'yu-ye-wang').
        - If int: Specifies the grid size N (defaults to 'modified shepp-logan').
        - If ndarray: A custom (M, 10) array defining ellipsoids.
    size_n : int, optional
        The grid size N (if the first argument was a string or array).
        
    Returns
    -------
    p : ndarray
        The 3D phantom volume of size (N, N, N).
    ellipses : ndarray
        The parameters used to generate the phantom.
    """
    
    # --- 1. Parse Inputs ---
    ellipses, n = _parse_inputs(size_or_def, size_n)
    
    # --- 2. Grid Generation ---
    # Create the grid from -1 to 1
    rng = np.linspace(-1.0, 1.0, n)
    # meshgrid in python (xy indexing) corresponds to standard cartesian
    X, Y, Z = np.meshgrid(rng, rng, rng, indexing='xy')
    
    # Flatten arrays to (1, N^3) for vectorized calculation
    # We stack them to create a (3, N^3) coordinate matrix
    coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    
    p = np.zeros(coords.shape[1])

    # --- 3. Ellipsoid Generation Loop ---
    for k in range(ellipses.shape[0]):
        # Extract parameters for the k-th ellipsoid
        A   = ellipses[k, 0]      # Amplitude
        asq = ellipses[k, 1]**2   # a^2
        bsq = ellipses[k, 2]**2   # b^2
        csq = ellipses[k, 3]**2   # c^2
        x0  = ellipses[k, 4]      # x offset
        y0  = ellipses[k, 5]      # y offset
        z0  = ellipses[k, 6]      # z offset
        
        # Euler angles in radians
        phi   = np.deg2rad(ellipses[k, 7])
        theta = np.deg2rad(ellipses[k, 8])
        psi   = np.deg2rad(ellipses[k, 9])
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        # Euler rotation matrix
        # (Using exact math from the MATLAB script)
        alpha = np.array([
            [cpsi*cphi - ctheta*sphi*spsi,   cpsi*sphi + ctheta*cphi*spsi,  spsi*stheta],
            [-spsi*cphi - ctheta*sphi*cpsi, -spsi*sphi + ctheta*cphi*cpsi,  cpsi*stheta],
            [stheta*sphi,                   -stheta*cphi,                   ctheta]
        ])
        
        # Apply rotation to coordinates
        # shape: (3, 3) @ (3, num_voxels) -> (3, num_voxels)
        coordp = alpha @ coords
        
        # Check inequality: (x-x0)^2/a^2 + (y-y0)^2/b^2 + (z-z0)^2/c^2 <= 1
        # Note: The original code applies x0, y0, z0 to the ROTATED coordinates.
        dist = ( (coordp[0, :] - x0)**2 / asq + 
                 (coordp[1, :] - y0)**2 / bsq + 
                 (coordp[2, :] - z0)**2 / csq )
        
        idx = dist <= 1
        p[idx] += A

    # --- 4. Reshape and Return ---
    p = p.reshape(n, n, n)
    return p, ellipses

def _parse_inputs(arg1, arg2):
    """Helper to process varargin logic."""
    n = 128 # Default size
    e = None
    
    default_map = {
        'shepp-logan': _get_shepp_logan(),
        'modified shepp-logan': _get_modified_shepp_logan(),
        'yu-ye-wang': _get_yu_ye_wang()
    }

    # Logic to handle argument combinations
    if arg1 is None:
        # phantom3d()
        e = default_map['modified shepp-logan']
        
    elif isinstance(arg1, str):
        # phantom3d('type', ...)
        key = arg1.lower()
        if key in default_map:
            e = default_map[key]
        else:
            raise ValueError(f"Unknown default phantom: {key}")
        
        if arg2 is not None:
            n = int(arg2)
            
    elif isinstance(arg1, (int, float, np.integer)):
        # phantom3d(128)
        n = int(arg1)
        e = default_map['modified shepp-logan']
        
    elif isinstance(arg1, np.ndarray):
        # phantom3d(custom_matrix, ...)
        e = arg1
        if arg2 is not None:
            n = int(arg2)
    else:
        raise ValueError("Invalid input arguments")
        
    return e, n

# --- Ellipsoid Definitions ---

def _get_shepp_logan():
    e = _get_modified_shepp_logan()
    # Override first row intensities based on original Shepp-Logan
    e[:, 0] = [1, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    return e

def _get_modified_shepp_logan():
    # Columns: A, a, b, c, x0, y0, z0, phi, theta, psi
    return np.array([
        [ 1,    .6900, .920, .810,    0,      0,      0,      0,      0,      0],
        [-.8,   .6624, .874, .780,    0,   -.0184,    0,      0,      0,      0],
        [-.2,   .1100, .310, .220,   .22,     0,      0,    -18,      0,     10],
        [-.2,   .1600, .410, .280,  -.22,     0,      0,     18,      0,     10],
        [ .1,   .2100, .250, .410,    0,     .35,   -.15,     0,      0,      0],
        [ .1,   .0460, .046, .050,    0,      .1,    .25,     0,      0,      0],
        [ .1,   .0460, .046, .050,    0,     -.1,    .25,     0,      0,      0],
        [ .1,   .0460, .023, .050,  -.08,  -.605,     0,      0,      0,      0],
        [ .1,   .0230, .023, .020,    0,   -.606,     0,      0,      0,      0],
        [ .1,   .0230, .046, .020,   .06,  -.605,     0,      0,      0,      0]
    ])

def _get_yu_ye_wang():
    return np.array([
        [ 1,    .6900, .920, .900,    0,      0,      0,      0,      0,      0],
        [-.8,   .6624, .874, .880,    0,      0,      0,      0,      0,      0],
        [-.2,   .4100, .160, .210,  -.22,     0,    -.25,   108,      0,      0],
        [-.2,   .3100, .110, .220,   .22,     0,    -.25,    72,      0,      0],
        [ .2,   .2100, .250, .500,    0,     .35,   -.25,     0,      0,      0],
        [ .2,   .0460, .046, .046,    0,      .1,   -.25,     0,      0,      0],
        [ .1,   .0460, .023, .020,  -.08,   -.65,   -.25,     0,      0,      0],
        [ .1,   .0460, .023, .020,   .06,   -.65,   -.25,    90,      0,      0],
        [ .2,   .0560, .040, .100,   .06,  -.105,    .625,   90,      0,      0],
        [-.2,   .0560, .056, .100,    0,     .100,   .625,    0,      0,      0]
    ])