import numpy as np
from pyevtk.hl import imageToVTK, gridToVTK
import glob
import os
from joblib import Parallel, delayed
from magnetic.gx_box import GXBox
from tqdm import tqdm
from magnetic.mathops import curl, angles, directions_closure


def save_scalar_data(s, scalar_name, filename):
    imageToVTK(filename, pointData={scalar_name: s})
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


def energy_density(bx, by, bz):
    e = np.power(bx, 2) + np.power(by, 2) + np.power(bz, 2)
    return e / 8. / np.pi


def free_energy(e_high, e_low, absolute=True):
    if absolute:
        return e_high - e_low
    else:
        return (e_high - e_low) / e_low


def central_derivative(files, timestep):
    reader = vtkStructuredPointsReader()
    reader.SetFileName(files[0])
    mesh = np.array(meshio.read(files[1]).point_data['Bnlfffe'])
    print(np.shape(mesh))
    return None


def folder_derivative(path, field_name, func=free_energy, filter='.vtr', n_jobs=8, timestep=720):
    """TIME DERIVATIVE OF DATA IN FOLDER. ASSUMING EQUIDISTANT FILES"""
    """
        PARAMETERS: timestep in seconds
    """
    all_files = sorted(files_list(path, filter))
    files_triplets = [all_files[i-1: i+2] for i in range(1, len(all_files) - 1)]
    central_derivative(files_triplets[0], timestep)
    return None


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


def box2npy(filename, field_name):
    target_dir, basename = prepare_target_name(filename, target_dir='npy_field')
    vx, vy, vz = field_from_box(filename)
    basename += '.npy'
    basename = field_name + '_' + basename
    vx = np.expand_dims(vx, axis=-1)
    vy = np.expand_dims(vy, axis=-1)
    vz = np.expand_dims(vz, axis=-1)
    v = np.concatenate([vx, vy, vz], axis=-1)
    np.save(os.path.join(target_dir, basename), v)
    return None


def box2curl2vtk(filename, field_name):
    """CONVERTS GX BOX TO VTK CURL"""
    target_dir, basename = prepare_target_name(filename, target_dir='vtk_curl')
    vx, vy, vz = field_from_box(filename)
    cx, cy, cz = curl(vx, vy, vz)
    average_angle = np.mean(directions_closure(angles(vx, vy, vz, cx, cy, cz)))
    print(f'Average angle between field and curl: {average_angle}')
    save_vector_data(cx, cy, cz, field_name, os.path.join(target_dir, basename))
    return None


def convert_folder(path, field_name, func=box2vtk, filter='.sav', n_jobs=8):
    """YOU SHOULD SPECIFY WHAT FUNCTION TO USE IN PARALLEL"""
    files = files_list(path, filter)
    Parallel(n_jobs=n_jobs)(delayed(func)(file, field)
                            for file, field in tqdm(zip(files, [field_name]*len(files))))
    return None


def files_list(path, filter):
    print(f'Start searching for files *{filter} in folder: {path}')
    files = glob.glob(f'{path}/*{filter}')
    print(f'Found {len(files)} files')
    return files

