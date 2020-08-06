from magnetic.sav2vtk import convert_folder, box2curl2vtk, box2npy, free_energy, source_points, regular_grid
import os


if __name__ == '__main__':
    folder = 'C:/AppsData/gx_out'
    # folder_derivative(folder, 'dB', filter='.vtk')
    # convert_folder(folder, 'Bpot', func=box2npy)
    # free_energy('C:\\AppsData\\gx_out\\npy_field', 'C:\\AppsData\\NLFFFE\\npy_field')
    # looptrace = 'C:\\AppsData\\NLFFFE\\BASEMAPS\\sav\\loops\\tracesAIA_94NORH_NLFFFE_170903_224642.dat'
    # save_file = os.path.join(os.path.dirname(looptrace), 'test_sources')
    # source_points(looptrace, save_file)
    regular_grid('/home/sunshine/data/regular_grid')
    #read_looptrace('C:\\AppsData\\NLFFFE\\BASEMAPS\\sav\\loops\\tracesAIA_94NORH_NLFFFE_170903_224642.dat')
