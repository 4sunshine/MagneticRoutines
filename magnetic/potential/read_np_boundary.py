# https://github.com/njuguoyang/magnetic_modeling_codes/blob/main/example/NLFFF_CAR_UNI/potential/potential_boundary/read_boundary.pro

import numpy as np


def read_boundary(npz_file_path, output_file='potential_boundary.dat'):
    # Parameters (same as in IDL code)
    nx = 244
    ny = 132
    arcsec2cm = 7.32825e7
    xc = 759.86469 * arcsec2cm
    yc = -345.34813 * arcsec2cm
    dx = 4.0 * 0.50 * arcsec2cm
    dy = dx
    
    # Load data from NPZ file (assuming it contains 'bx', 'by', 'bz' arrays)
    data = np.load(npz_file_path)
    bz = data['bz']  # Assuming the dictionary key is 'bz'
    
    # Trim the array (same as IDL's Bz[2:nx-3,2:ny-3])
    bz_trimmed = bz[2:-3, 2:-3]
    nx1, nx2 = bz_trimmed.shape
    
    # Write to binary file
    with open(output_file, 'wb') as f:
        # Write integers (4 bytes each)
        np.array([nx1], dtype=np.int32).tofile(f)
        np.array([nx2], dtype=np.int32).tofile(f)
        
        # Write doubles (8 bytes each)
        np.array([xc], dtype=np.float64).tofile(f)
        np.array([yc], dtype=np.float64).tofile(f)
        np.array([dx], dtype=np.float64).tofile(f)
        np.array([dy], dtype=np.float64).tofile(f)
        
        # Write Bz array as doubles
        bz_trimmed.astype(np.float64).tofile(f)
    
    # Print information
    print(f'Bz range (Gauss): {bz_trimmed.min()}, {bz_trimmed.max()}')
    print('Computation domain for potential field:')
    print(f'nx1,nx2: {nx1}, {nx2}')
    print(f'xc,yc (cm): {xc}, {yc}')
    print(f'dx,dy (cm): {dx}, {dy}')
    
    x1 = xc - nx1 * dx / 2
    x2 = xc + nx1 * dx / 2
    y1 = yc - nx2 * dy / 2
    y2 = yc + nx2 * dy / 2
    
    print('x,y, and z range (10 Mm):')
    print(f'        xprobmin1= {x1 * 1.e-9:.12f}d0')
    print(f'        xprobmax1= {x2 * 1.e-9:.12f}d0')
    print(f'        xprobmin2= {y1 * 1.e-9:.12f}d0')
    print(f'        xprobmax2= {y2 * 1.e-9:.12f}d0')
    print(f'        xprobmin3= {0.0 * 1.e-9 + 0.1:.12f}d0')  # to lift the domain 1 Mm above 0
    print(f'        xprobmax3= {(y2 - y1) * 1.e-9:.12f}d0')

# Example usage:
# read_boundary('your_file.npz')