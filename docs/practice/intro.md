# Solar atmosphere magnetic field and electric currents analysis

## Short introduction  

Big goal:
https://arxiv.org/abs/2310.19617  

Solar activity overview:  
https://solarmonitor.org/index.php  


## Analysis tools  

Paraview for visualization:  
https://www.paraview.org/download/  

### Data sample:  

Extrapolated magnetic field:  
https://disk.yandex.ru/d/ce7rYOAz4zy3lA

Magnetic field basemap:  
https://disk.yandex.ru/d/C_7UwSlM9suKeg

## Initial Task  

Download data sample, and visualize it with Paraview:  

1. Open extrapolated magnetic field `.vtr` file.  
2. Apply `ResampleToImage` filter to it.  
3. Open basemap field `.vtk` file.  
4. Analyze magnetic field lines in solar atmosphere with `StreamTracer` filter.


