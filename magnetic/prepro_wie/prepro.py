import numpy as np


def laplace(field):
    """
    (Five-point stencil) finite-difference formula

    Compute the discrete Laplace operator for a 2D field.
    
    Parameters:
    -----------
    field : numpy.ndarray
        2D input array
        
    Returns:
    --------
    numpy.ndarray
        Laplace operator applied to the input field
    """
    return (-4 * field + np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) 
            + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))


def prepro(Bx, By, Bz, mu3, mu4, nx, ny, nz, noprint=False):
    """
    Preprocessing of vector magnetograms for a nonlinear force-free reconstruction.
    
    Parameters:
    -----------
    Bx, By, Bz : float
        [x_size, y_size] B field
    mu3 : float
        Weight for observations term
    mu4 : float
        Weight for smoothing term
    nx, ny, nz : int
        Dimensions of the grid
    noprint : bool, optional
        If True, suppress progress output
        
    Returns:
    --------
    tuple
        (L1+L2, L3, L4) where these are the different components of the functional
    """

    # Check if magnetic field components exist and have same shape
    if Bx.shape != By.shape or Bx.shape != Bz.shape:
        raise ValueError("2D-fields Bx, By, Bz must exist and have same size.")
    
    if not noprint:
        print("Starting preprocessing...")
    
    # Normalize coordinates
    x1 = np.linspace(0, 1, nx)
    y1 = np.linspace(0, 1, ny)
    z1 = np.linspace(0, 1, nz)
    
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dxdy = dx * dy
    
    # Create meshgrid
    x, y = np.meshgrid(x1, y1, indexing='ij')

    # Normalize magnetic field
    bave2d = np.mean(np.sqrt(Bx**2 + By**2 + Bz**2))
    Bx = Bx / bave2d
    By = By / bave2d
    Bz = Bz / bave2d
    
    # Save original values
    Bxo = Bx.copy()
    Byo = By.copy()
    Bzo = Bz.copy()
    
    # Working copies
    Bx1 = Bx.copy()
    By1 = By.copy()
    Bz1 = Bz.copy()
    
    # Weight parameters
    mu1 = 1.0  # Force
    mu2 = mu1  # Torque
    
    # Scale factors
    fac1 = 10.0
    mu1 = mu1 / fac1
    mu2 = mu2 / fac1
    mu3 = mu3 / fac1
    mu4 = mu4 / fac1
    
    if not noprint:
        print(f"mu1, mu2, mu3, mu4: {mu1}, {mu2}, {mu3}, {mu4}")
    
    # Initial calculations
    emag = np.sum(Bx**2 + By**2 + Bz**2)
    # emag2 = emag * emag
    
    r = np.sqrt(x**2 + y**2)
    ihelp = np.sum(r * (Bx**2 + By**2 + Bz**2))
    # ihelp2 = ihelp * ihelp
    
    # Optimization loop
    it = -1
    L = 10
    dL = 1.0
    
    if not noprint:
        print("dL, eps_force, eps_torque, eps_smooth:")
        print("it, L1, L2, L3, L4:")
    
    while it < 2000 and dL > 1.0e-4:
        it += 1
        
        # Force terms
        term1a = np.sum(Bx * Bz) / emag
        term1b = np.sum(By * Bz) / emag
        term1c = (np.sum(Bz**2) - np.sum(Bx**2 + By**2)) / emag
        
        # Torque terms
        term2a = (np.sum(x * Bz**2) - np.sum(x * (Bx**2 + By**2))) / ihelp
        term2b = (np.sum(y * Bz**2) - np.sum(y * (Bx**2 + By**2))) / ihelp
        term2c = (np.sum(y * Bx * Bz) - np.sum(x * By * Bz)) / ihelp
        
        # Observation terms
        term3a = (Bx1 - Bxo)**2 / emag
        term3b = (By1 - Byo)**2 / emag
        term3c = (Bz1 - Bzo)**2 / emag

        # Smoothing terms
        term4a = 2.0 * (-4.0 * laplace(Bx) + laplace(np.roll(Bx, 1, axis=0)) +
                        laplace(np.roll(Bx, -1, axis=0)) + laplace(np.roll(Bx, 1, axis=1)) +
                        laplace(np.roll(Bx, -1, axis=1)))
        
        term4b = 2.0 * (-4.0 * laplace(By) + laplace(np.roll(By, 1, axis=0)) +
                        laplace(np.roll(By, -1, axis=0)) + laplace(np.roll(By, 1, axis=1)) +
                        laplace(np.roll(By, -1, axis=1)))
        
        term4c = 2.0 * (-4.0 * laplace(Bz) + laplace(np.roll(Bz, 1, axis=0)) +
                        laplace(np.roll(Bz, -1, axis=0)) + laplace(np.roll(Bz, 1, axis=1)) +
                        laplace(np.roll(Bz, -1, axis=1)))

        eps_smooth = (np.sum(np.abs(laplace(Bx))) + np.sum(np.abs(laplace(By))) + 
                     np.sum(np.abs(laplace(Bz))))
        eps_smooth = eps_smooth * dxdy / np.sqrt(emag)

        # Calculate functional components
        L1 = term1a**2 + term1b**2 + term1c**2
        eps_force = np.abs(term1a) + np.abs(term1b) + np.abs(term1c)
        
        L2 = term2a**2 + term2b**2 + term2c**2
        eps_torque = np.abs(term2a) + np.abs(term2b) + np.abs(term2c)
        
        L12 = L1 + L2
        L3 = np.sum(term3a + term3b + term3c) / emag
        L4 = dxdy * np.sum(laplace(Bx)**2 + laplace(By)**2 + laplace(Bz)**2) / emag
        L = L1 + L2 + L3 + L4
        
        if it > 0:
            dL = np.abs(L12 - oldL12) / L12 + np.abs(L3 - oldL3) / L3 + np.abs(L4 - oldL4) / L4

        if it % 50 == 0 and not noprint:
            print(dL, eps_force, eps_torque, eps_smooth)
        
        oldL12 = L12
        oldL3 = L3
        oldL4 = L4
        
        if it % 50 == 0 and not noprint:
            print(it, L1, L2, L3, L4)
        
        # Update magnetic field components
        Bx1 = (Bx + mu1 * (-2.0 * term1a * Bz + 4.0 * term1c * Bx) - mu3 * 2.0 * (Bx - Bxo) -
               mu4 * term4a + mu2 * (4.0 * term2a * x * Bx + 4.0 * term2b * y * Bx - 
                                     2.0 * term2c * y * Bz))
        
        By1 = (By + mu1 * (-2.0 * term1b * Bz + 4.0 * term1c * By) - mu3 * 2.0 * (By - Byo) -
               mu4 * term4b + mu2 * (4.0 * term2a * x * By + 4.0 * term2b * y * By + 
                                     2.0 * term2c * x * Bz))
        
        Bz1 = Bz - mu3 * 2.0 * (Bz - Bzo) - mu4 * term4c
        
        Bx = Bx1
        By = By1
        Bz = Bz1

    # Restore original scaling
    Bx = Bx * bave2d
    By = By * bave2d
    Bz = Bz * bave2d

    if not noprint:
        print("Correct magnetogram finished")
    
    return (Bx, By, Bz), (L1 + L2, L3, L4)
