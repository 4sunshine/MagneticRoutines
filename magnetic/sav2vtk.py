import numpy as np
from pyevtk.hl import imageToVTK, gridToVTK
import glob
import os
from joblib import Parallel, delayed
from magnetic.gx_box import GXBox
from tqdm import tqdm
from magnetic.mathops import curl, angles, directions_closure


def save_scalar_data(vx, vy, vz, vector_name, filename):
    """NEED TO RE-IMPLEMENT THIS FUNCTION"""
    assert np.shape(vx) == np.shape(vy) == np.shape(vz)
    imageToVTK(filename, cellData={f'{vector_name} x': vx, f'{vector_name} y': vy,
               f'{vector_name} z': vz})
    return None


def save_vector_data(vx, vy, vz, vector_name, filename, origin=(0., 0., 0.), spacing=(1., 1., 1.)):
    """CORRECTLY SAVES VECTOR DATA LIKE IN IDL's: sav2vtk.pro"""
    assert np.shape(vx) == np.shape(vy) == np.shape(vz)
    X = np.arange(origin[0], np.shape(vx)[0], spacing[0], dtype='float64')
    Y = np.arange(origin[1], np.shape(vx)[1], spacing[1], dtype='float64')
    Z = np.arange(origin[2], np.shape(vx)[2], spacing[2], dtype='float64')
    gridToVTK(filename, X, Y, Z, pointData={vector_name: (vx, vy, vz)})
    return None


def field_from_box(filename):
    box = GXBox(filename)
    bx = np.array(box.bx, dtype='float64')
    by = np.array(box.by, dtype='float64')
    bz = np.array(box.bz, dtype='float64')
    return bx, by, bz


def prepare_target_name(filename, target_dir='target'):
    cd = os.path.dirname(filename)
    basename = os.path.basename(filename)
    basename = basename.split('.')[:-1]
    basename = ''.join(basename)  # + '.vtk'
    target_dir = os.path.join(cd, target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir, basename


def box2vtk(filename, field_name):
    target_dir, basename = prepare_target_name(filename, target_dir='vtk_field')
    vx, vy, vz = field_from_box(filename)
    save_vector_data(vx, vy, vz, field_name, os.path.join(target_dir, basename))
    return None


def box2curl2vtk(filename, field_name):
    target_dir, basename = prepare_target_name(filename, target_dir='vtk_curl')
    vx, vy, vz = field_from_box(filename)
    cx, cy, cz = curl(vx, vy, vz)
    average_angle = np.mean(directions_closure(angles(vx, vy, vz, cx, cy, cz)))
    print(f'Average angle between field and curl: {average_angle}')
    save_vector_data(cx, cy, cz, field_name, os.path.join(target_dir, basename))
    return None


def convert_folder(path, field_name, filter='.sav', n_jobs=8):
    print(f'Start converting folder: {path}')
    files = glob.glob(f'{path}/*{filter}')
    print(f'Found {len(files)} files')
    Parallel(n_jobs=n_jobs)(delayed(box2curl2vtk)(file, field)
                            for file, field in tqdm(zip(files, [field_name]*len(files))))
    return None

