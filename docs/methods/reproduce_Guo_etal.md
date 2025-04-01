# Data-driven Modeling of a Coronal Magnetic Flux Rope: From Birth to Death Reproduction

## Steps to Reproduce

### 1. Acquire Flare Coordinates & Track Pre-History
**Time Range:**  
`2017-09-03 09:00:04` to `2017-09-06 12:00:04`

**Coordinates:**  
- **Start Position:**
  - CRLON = `115.628°`
  - CRLAT = `-8.0°`
  - X = `-129.205″`
  - Y = `-249.736″`

- **End Position:**
  - CRLON = `115.941°`
  - CRLAT = `-8.0°`
  - X = `525.367″`
  - Y = `-231.136″`

---

### 2. Reconstruct Magnetic Field BBOX Series
**Parameters:**
- Resolution: `dx_km = 367.084` (km/pix)
- Grid Size: `size_pix = [560, 400, 560]`

**Method:**  
- 720s HMI vector data
- CEA_PROJECTION
- Tool: [GXBox_prep](https://github.com/Sergey-Anfinogentov/GXBox_prep)

---

### 3. Calculate Plasma Velocity (DAVE4VM)
**Parameters:**
- `window_size = 32`

**Implementation:**
- IDL Source Code: [pydave4vm](https://github.com/Chicrala/pydave4vm/blob/master/pydave4vm/DAVE4VM-CORE.tar.gz)

**Encountered Issue:**
`% DAVE4VM: The input images MAG.Bx are pathological or the window
% DAVE4VM: size SIGMA is too small. The aperture problem cannot be solved`  

This error occurs in many setups, preventing successful execution.

### 4. Run MHD Simulation (MPI-AMRVAC)
**Parameters:**
- TBD  


## Questions:

Do you have an open-source code to:

1. Get and track HMI/SDO magnetograms with bbox creation? *I have one, but I need to compare methodologies, and verify myself*

2. Obtain photosphere plasma velocity? *I had not run DAVE4VM succesfully yet.* 
