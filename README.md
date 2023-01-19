# SCRIPTS FOR SOLAR ACTIVE REGIONS MAGNETIC FIELD ANALYSIS

## IDL scripts  
### Creation of GX simulator-compatible boxes  
Create *New project* in IDL. Set local directory of this repo as a project's folder. 
Compile created project. Edit required parameters of `main.pro` and run it. Boxes with potential field 
extrapolation will be created.  
To make linear force-free extrapolation run:  
`idl_magnetic/multi_nlfffe.pro`  
To save data cube as `.vtk` file run:  
`idl_magnetic/sav2vtk.pro`  
Basemaps can be imported and created using `idl_magnetic/makebasemap.pro`.  

*Note that python conversion scripts run more effectively in terms of speed and result files sizes.
