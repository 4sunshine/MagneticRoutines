from magnetic.sav2vtk import convert_folder, box2curl2vtk, folder_derivative, box2npy


if __name__ == '__main__':
    folder = 'C:/AppsData/gx_out'
    # folder_derivative(folder, 'dB', filter='.vtk')
    convert_folder(folder, 'Bpot', func=box2npy)

