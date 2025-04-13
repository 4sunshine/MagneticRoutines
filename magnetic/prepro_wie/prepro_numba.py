import numpy as np
from numba import njit, prange

@njit(parallel=True)
def laplace_numba(field):
    """
    Optimized Laplace operator using numba for parallel computation.
    """
    result = np.empty_like(field)
    nx, ny = field.shape
    
    for i in prange(nx):
        for j in prange(ny):
            # Handle boundary conditions with modulo operations
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            result[i, j] = (-4 * field[i, j] + field[im1, j] + field[ip1, j] + 
                            field[i, jm1] + field[i, jp1])
    
    return result

def prepro(Bx, By, Bz, mu3, mu4, nx, ny, nz, noprint=False):
    """
    Optimized preprocessing of vector magnetograms with:
    - Numba JIT compilation
    - Vectorized operations
    - Reduced memory allocations
    - Parallel computation
    - Optimized boundary handling
    """
    # Input validation
    if Bx.shape != By.shape or Bx.shape != Bz.shape:
        raise ValueError("2D-fields Bx, By, Bz must exist and have same size.")
    
    if not noprint:
        print("Starting preprocessing (optimized version)...")
    
    # Pre-compute constants
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dxdy = dx * dy
    
    # Normalize magnetic field (vectorized)
    bave2d = np.mean(np.sqrt(Bx**2 + By**2 + Bz**2))
    Bx_norm = Bx / bave2d
    By_norm = By / bave2d
    Bz_norm = Bz / bave2d
    
    # Save original values
    Bxo = Bx_norm.copy()
    Byo = By_norm.copy()
    Bzo = Bz_norm.copy()
    
    # Working copies
    Bx1 = Bx_norm.copy()
    By1 = By_norm.copy()
    Bz1 = Bz_norm.copy()
    
    # Weight parameters
    mu1 = 1.0 / 10.0  # Combined Force and scaling
    mu2 = mu1         # Torque
    mu3 = mu3 / 10.0  # Observation
    mu4 = mu4 / 10.0  # Smoothing
    
    if not noprint:
        print(f"mu1, mu2, mu3, mu4: {mu1}, {mu2}, {mu3}, {mu4}")
    
    # Precompute grid and initial calculations
    x = np.linspace(0, 1, nx)[:, None]  # Column vector
    y = np.linspace(0, 1, ny)[None, :]  # Row vector
    r = np.sqrt(x**2 + y**2)
    
    # Pre-allocate arrays for terms to avoid repeated allocation
    term1a_arr = np.empty_like(Bx_norm)
    term1b_arr = np.empty_like(Bx_norm)
    term1c_arr = np.empty_like(Bx_norm)
    term2a_arr = np.empty_like(Bx_norm)
    term2b_arr = np.empty_like(Bx_norm)
    term2c_arr = np.empty_like(Bx_norm)
    
    # Initialize variables for the loop
    it = -1
    L = 10
    dL = 1.0
    oldL12, oldL3, oldL4 = 0, 0, 0
    
    if not noprint:
        print("dL, eps_force, eps_torque, eps_smooth:")
        print("it, L1, L2, L3, L4:")
    
    # Main optimization loop
    while it < 2000 and dL > 1.0e-4:
        it += 1
        
        # Compute emag and ihelp once per iteration
        emag = np.sum(Bx_norm**2 + By_norm**2 + Bz_norm**2)
        ihelp = np.sum(r * (Bx_norm**2 + By_norm**2 + Bz_norm**2))
        
        # Force terms (vectorized)
        BxBz = Bx_norm * Bz_norm
        ByBz = By_norm * Bz_norm
        Bz2 = Bz_norm**2
        Bx2_plus_By2 = Bx_norm**2 + By_norm**2
        
        term1a = np.sum(BxBz) / emag
        term1b = np.sum(ByBz) / emag
        term1c = (np.sum(Bz2) - np.sum(Bx2_plus_By2)) / emag
        
        # Torque terms (vectorized)
        xBz2 = x * Bz2
        xBx2_plus_By2 = x * Bx2_plus_By2
        yBz2 = y * Bz2
        yBx2_plus_By2 = y * Bx2_plus_By2
        yBxBz = y * BxBz
        xByBz = x * ByBz
        
        term2a = (np.sum(xBz2) - np.sum(xBx2_plus_By2)) / ihelp
        term2b = (np.sum(yBz2) - np.sum(yBx2_plus_By2)) / ihelp
        term2c = (np.sum(yBxBz) - np.sum(xByBz)) / ihelp
        
        # Observation terms (vectorized)
        term3a = (Bx1 - Bxo)**2 / emag
        term3b = (By1 - Byo)**2 / emag
        term3c = (Bz1 - Bzo)**2 / emag
        
        # Smoothing terms using optimized laplace function
        laplace_Bx = laplace_numba(Bx_norm)
        laplace_By = laplace_numba(By_norm)
        laplace_Bz = laplace_numba(Bz_norm)
        
        eps_smooth = (np.sum(np.abs(laplace_Bx)) + np.sum(np.abs(laplace_By)) + 
                     np.sum(np.abs(laplace_Bz)))
        eps_smooth = eps_smooth * dxdy / np.sqrt(emag)
        
        # Calculate functional components
        L1 = term1a**2 + term1b**2 + term1c**2
        eps_force = np.abs(term1a) + np.abs(term1b) + np.abs(term1c)
        
        L2 = term2a**2 + term2b**2 + term2c**2
        eps_torque = np.abs(term2a) + np.abs(term2b) + np.abs(term2c)
        
        L12 = L1 + L2
        L3 = np.sum(term3a + term3b + term3c) / emag
        L4 = dxdy * np.sum(laplace_Bx**2 + laplace_By**2 + laplace_Bz**2) / emag
        
        if it > 0:
            dL = (np.abs(L12 - oldL12) / L12 + 
                  np.abs(L3 - oldL3) / L3 + 
                  np.abs(L4 - oldL4) / L4)
        
        if it % 50 == 0 and not noprint:
            print(dL, eps_force, eps_torque, eps_smooth)
            print(it, L1, L2, L3, L4)
        
        oldL12 = L12
        oldL3 = L3
        oldL4 = L4
        
        # Update magnetic field components (vectorized)
        Bx1 = (Bx_norm + mu1 * (-2.0 * term1a * Bz_norm + 4.0 * term1c * Bx_norm) - 
               mu3 * 2.0 * (Bx_norm - Bxo) - mu4 * 2.0 * (-4.0 * laplace_Bx + 
               laplace_numba(np.roll(Bx_norm, 1, axis=0)) + laplace_numba(np.roll(Bx_norm, -1, axis=0)) + 
               laplace_numba(np.roll(Bx_norm, 1, axis=1)) + laplace_numba(np.roll(Bx_norm, -1, axis=1))) + 
               mu2 * (4.0 * term2a * x * Bx_norm + 4.0 * term2b * y * Bx_norm - 2.0 * term2c * y * Bz_norm))
        
        By1 = (By_norm + mu1 * (-2.0 * term1b * Bz_norm + 4.0 * term1c * By_norm) - 
               mu3 * 2.0 * (By_norm - Byo) - mu4 * 2.0 * (-4.0 * laplace_By + 
               laplace_numba(np.roll(By_norm, 1, axis=0)) + laplace_numba(np.roll(By_norm, -1, axis=0)) + 
               laplace_numba(np.roll(By_norm, 1, axis=1)) + laplace_numba(np.roll(By_norm, -1, axis=1))) + 
               mu2 * (4.0 * term2a * x * By_norm + 4.0 * term2b * y * By_norm + 2.0 * term2c * x * Bz_norm))
        
        Bz1 = Bz_norm - mu3 * 2.0 * (Bz_norm - Bzo) - mu4 * 2.0 * (-4.0 * laplace_Bz + 
               laplace_numba(np.roll(Bz_norm, 1, axis=0)) + laplace_numba(np.roll(Bz_norm, -1, axis=0)) + 
               laplace_numba(np.roll(Bz_norm, 1, axis=1)) + laplace_numba(np.roll(Bz_norm, -1, axis=1)))
        
        # Update for next iteration
        Bx_norm = Bx1
        By_norm = By1
        Bz_norm = Bz1
    
    # Restore original scaling
    Bx_result = Bx_norm * bave2d
    By_result = By_norm * bave2d
    Bz_result = Bz_norm * bave2d

    if not noprint:
        print("Correct magnetogram finished")
    
    return (Bx_result, By_result, Bz_result), (L12, L3, L4)