# trace generated using paraview version 5.11.0-RC1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
import os
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

FILE_TO_OPEN = "D:\\AppsData\\NLFFFE\\vtk\\NORH_NLFFFE_170903_224642.vtr"
MAP_FILE_TO_OPEN = "D:\\AppsData\\NLFFFE\\BASEMAPS\\vtk\\AIA_131NORH_NLFFFE_170903_224642.vtk"

def get_registration_name(name, prefix=""):
    _registration_name = os.path.basename(name)
    _registration_name = "".join(_registration_name.split(".")[:-1])
    if prefix:
        _registration_name = prefix + "_" + _registration_name

    return _registration_name.capitalize()

# create a new 'XML Rectilinear Grid Reader'
b_field = XMLRectilinearGridReader(registrationName=get_registration_name(FILE_TO_OPEN, "B_field"), FileName=[FILE_TO_OPEN])

# Properties modified on b_field
b_field.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
b_fieldDisplay = Show(b_field, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
b_fieldDisplay.Representation = 'Outline'

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Resample To Image'
b = ResampleToImage(registrationName=get_registration_name(FILE_TO_OPEN, "B"), Input=b_field)

# show data in view
bDisplay = Show(b, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
bDisplay.Representation = 'Outline'

# hide data in view
Hide(b_field, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(bDisplay, ('POINTS', 'B nlfffe', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
bDisplay.RescaleTransferFunctionToDataRange(True, True)

# change representation type
bDisplay.SetRepresentationType('Volume')

# rescale color and/or opacity maps used to include current data range
bDisplay.RescaleTransferFunctionToDataRange(True, False)

# get color transfer function/color map for 'Bnlfffe'
bnlfffeLUT = GetColorTransferFunction('Bnlfffe')

# get opacity transfer function/opacity map for 'Bnlfffe'
bnlfffePWF = GetOpacityTransferFunction('Bnlfffe')

# get 2D transfer function for 'Bnlfffe'
bnlfffeTF2D = GetTransferFunction2D('Bnlfffe')

# Rescale transfer function
bnlfffeLUT.RescaleTransferFunction(0.0, 2000.0)

# Rescale transfer function
bnlfffePWF.RescaleTransferFunction(0.0, 2000.0)

# Rescale 2D transfer function
bnlfffeTF2D.RescaleTransferFunction(0.0, 2000.0, 0.0, 1.0)

# create a new 'Legacy VTK Reader'
basemap = LegacyVTKReader(registrationName=get_registration_name(MAP_FILE_TO_OPEN, "Basemap"), FileNames=[MAP_FILE_TO_OPEN])

# show data in view
basemapDisplay = Show(basemap, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
basemapDisplay.Representation = 'Slice'

# show color bar/color legend
basemapDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'variable'
variableLUT = GetColorTransferFunction('variable')

# get opacity transfer function/opacity map for 'variable'
variablePWF = GetOpacityTransferFunction('variable')

# get 2D transfer function for 'variable'
variableTF2D = GetTransferFunction2D('variable')
