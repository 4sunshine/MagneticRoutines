# Magneto-Frictional Method for Relaxing 3D Magnetic Fields to Force-Free Fields

## Overview
The **magneto-frictional method** developed by **Chun Xia** and **Yang Guo** is implemented in the **[MPI-AMRVAC](https://github.com/amrvac/amrvac)** code. It is designed to relax a 3D magnetic field to a **nonlinear force-free field (NLFFF)**, which is particularly useful for modeling the solar corona. The method was modulized by **Chun Xia** on **04.10.2017**, making it easier to integrate into the MPI-AMRVAC framework.

---

## Key Details

### Purpose
The magneto-frictional method is used to:
- Relax a 3D magnetic field to a **force-free state** (where \(\mathbf{J} \times \mathbf{B} = 0\)).
- Model the solar corona, where the magnetic field is often in a force-free state.

### Implementation
- The method is implemented in the **[MPI-AMRVAC](https://github.com/amrvac/amrvac)** code, a parallel, adaptive mesh refinement (AMR) framework for solving magnetohydrodynamic (MHD) equations.
- It supports **Cartesian** and **spherical coordinates**, as well as **uniform** and **block-adaptive octree grids**.
- High spatial resolution and large field-of-view simulations are enabled, making it suitable for observation-constrained extrapolations.

### Modularization
- The method was modulized by **Chun Xia** on **04.10.2017**, simplifying its integration into the MPI-AMRVAC framework.

---

## Usage in MPI-AMRVAC

To use the magneto-frictional method in MPI-AMRVAC, the following parameters need to be set in the `amrvac.par` file:

### Key Parameters
- **Time Stepping**:  
  ```plaintext
  time_stepper='onestep'  # or 'twostep', 'threestep'
  ```
- **Flux Method**:  
  ```plaintext
  flux_method=13*'cd4'    # or 'tvdlf', 'fd'
  ```
- **Limiter**:  
  ```plaintext
  limiter=13*'koren'      # or 'vanleer', 'cada3', 'mp5'
  ```
- **Magnetofriction Activation**:  
  ```plaintext
  mhd_magnetofriction=.true.
  ```
- **Iteration Parameters**:  
  ```plaintext
  mf_it_max=60000         # maximum iteration number
  mf_ditsave=20000        # iteration interval for data output
  mf_cc=0.3               # stability coefficient
  mf_cy=0.2               # frictional velocity coefficient
  mf_cdivb=0.01           # divergence cleaning coefficient
  ```

For more details on parameter settings, refer to the **[MPI-AMRVAC Documentation](https://amrvac.org/doc/)**.

---

## Applications

The magneto-frictional method has been applied to various solar phenomena, including:

### 1. **Filament Modeling**
- Used to model the magnetic structure of solar filaments, which are often associated with flux ropes. For more details, see [Xia & Guo (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...851...75X/abstract).

### 2. **Active Region Modeling**
- Applied to reconstruct the magnetic field of solar active regions, providing insights into the onset of solar flares and coronal mass ejections (CMEs). For more details, see [Xia et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...853...49X/abstract).

---

## Overview Conclusion

The magneto-frictional method developed by **Chun Xia** and **Yang Guo** is a powerful tool for modeling the solar corona's magnetic field. Its implementation in **[MPI-AMRVAC](https://github.com/amrvac/amrvac)** allows for high-resolution simulations of complex magnetic structures, making it a valuable resource for solar physics research.

For more details, refer to the original papers and the **[MPI-AMRVAC Documentation](https://amrvac.org/doc/)**.


## Initial conditions  

To insert **initial magnetic field conditions** from a **3D VTK file** into the **MPI-AMRVAC** framework for use with the **magneto-frictional method**, you need to follow these steps:

---

## 1. **Prepare the VTK File**
Ensure your VTK file contains the **magnetic field components** (`Bx`, `By`, `Bz`) in a format compatible with MPI-AMRVAC. The VTK file should be structured as follows:
- **Data Format**: Use **structured grid** or **unstructured grid** format.
- **Field Components**: Include the magnetic field components as **point data** or **cell data**.
- **Units**: Ensure the magnetic field is in the correct units (e.g., Gauss or Tesla).

Example VTK file structure:
```plaintext
# vtk DataFile Version 3.0
3D Magnetic Field Data
ASCII
DATASET STRUCTURED_GRID
DIMENSIONS NX NY NZ
POINTS NX*NY*NZ float
x1 y1 z1
x2 y2 z2
...
POINT_DATA NX*NY*NZ
VECTORS B float
Bx1 By1 Bz1
Bx2 By2 Bz2
...
```

---

## 2. **Convert VTK to MPI-AMRVAC Format**
MPI-AMRVAC uses its own **binary format** for initial conditions. You need to convert the VTK file into this format. Follow these steps:

### a. **Install Python Tools**
MPI-AMRVAC provides Python tools for data conversion. Install them using:
https://amrvac.org/md_doc_python_setup.html  


### b. **Convert VTK to AMRVAC Format**
Use the `amrvac_pytools` library to convert the VTK file. Here’s an example script:

```python
from amrvac_pytools.datfiles.reader import load_datfile
from amrvac_pytools.vtkfiles.writer import write_vtkfile
import numpy as np

# Load the VTK file
from vtk import vtkStructuredGridReader
reader = vtkStructuredGridReader()
reader.SetFileName("input.vtk")
reader.Update()
data = reader.GetOutput()

# Extract magnetic field components
points = data.GetPoints()
Bx = data.GetPointData().GetArray("B").GetComponent(0)
By = data.GetPointData().GetArray("B").GetComponent(1)
Bz = data.GetPointData().GetArray("B").GetComponent(2)

# Create AMRVAC-compatible arrays
nx, ny, nz = data.GetDimensions()
Bx_arr = np.reshape(Bx, (nx, ny, nz))
By_arr = np.reshape(By, (nx, ny, nz))
Bz_arr = np.reshape(Bz, (nx, ny, nz))

# Save as AMRVAC datfile
from amrvac_pytools.datfiles.writer import write_datfile
write_datfile("output.dat", Bx_arr, By_arr, Bz_arr)
```

---

## 3. **Modify the `amrvac.par` File**
Update the `amrvac.par` file to load the initial magnetic field from the converted file. Add the following parameters:

```plaintext
&filelist
  base_filename = 'output'  # Name of the datfile (without extension)
  type_filelog = 'default'
  type_datfile = 'binary'   # Binary format for the datfile
/
```

---

## 4. **Set Up the Initial Conditions in MPI-AMRVAC**
In the `mod_usr.t` file (user-defined module), specify how to load the initial magnetic field. For example:

```fortran
subroutine initglobaldata_usr
  use mod_global_parameters
  implicit none

  ! Load the initial magnetic field from the datfile
  call init_one_grid(igrid)
end subroutine initglobaldata_usr
```

---

## 5. **Run the Simulation**
Compile and run MPI-AMRVAC with the updated `amrvac.par` file:
```bash
mpirun -np 4 ./amrvac -i amrvac.par
```

---

## 6. **Verify the Initial Conditions**
After running the simulation, check the output files to ensure the magnetic field has been correctly initialized. You can use the `amrvac_pytools` library to visualize the results:
```python
from amrvac_pytools.datfiles.reader import load_datfile
data = load_datfile("output0000.dat")
print(data.keys())  # Check available fields
```

---

## Additional Resources
- **[MPI-AMRVAC Documentation](https://amrvac.org/doc/)**
- **[AMRVAC Python Tools](https://github.com/amrvac/amrvac_pytools)**
