# SCRIPTS FOR SOLAR ACTIVE REGIONS MAGNETIC FIELD ANALYSIS

## :sun_with_face::sun_with_face::sun_with_face: Active Regions Dynamics Analysis Pipeline :sun_with_face::sun_with_face::sun_with_face:  
  
### Step 1. Getting the data for the Period  
Add your JSOC_EMAIL to the ENV in bash:  
`export JSOC_EMAIL="your@email.com"`  
Modify parameters like analysis time & download dir in the script    
`python magnetic/jsoc_query.py`  
Then run it in a `screen` interface. SDO/HMI 720s files will be downloaded in the output_dir.  
When at least a part of files downloaded, run this script again to prepare `txt` files, which are necessary for IDL.  
  
### Step 2. Manually setting parameters of analysis with IDL  
Create a project in IDL from `mmc` folder of this repo and compile it.  
All following scripts are related to `mmc/vmf` subdir.  
  
#### Select a region for analysis  
This steps should be done for the First and Last moments in time: for at least two `EVENT_TIME`'s.
`prepare_hmi, DATA_DIR, EVENT_TIME, REFERENCE_TIME, SAVE_ALL_DIR`  
where  
`DATA_DIR` -- a folder with downloaded SDO/HMI data;  
`EVENT_TIME`  -- an observation time in format `20170903_090000_TAI`;  
`REFERENCE_TIME`  -- time of the first observation in the same format;  
`SAVE_ALL_DIR`  -- output directory, where all data will be stored.  
  
:pushpin: Fix the observed center of the Sun
```IDL  
;FIND CENTER HERE
dx0 = 0.5*(956.5 - 949.5)
dy0 = 0.5*(950.0 - 953.0)  
```  
via plotting in ranges dx, dy and vice-versa `~ +/- (940 -- 960)` and `~ (-10 -- 10)`.  

```IDL
; DRAW A SUBREGION HERE. USED FOR SOLAR LIMB COORDINATES FINDING
sub_map,mapbz,smapbz,xrange=[-250,50.0],yrange=[-400,-100.0]
```  
After all you also should select a region of analysis here.  

:pushpin: Look at the vector fields directions  
```IDL
plot_vmap,/over,smapbx,smapby,mapbz=smapbz,limit=180,scale=0.012,iskip=15,jskip=15,$
  v_color=255,axis_color=0,/Nolabels,v_thick=2.0 ;,/No_arrow_head  ;,/Noaxis
``` 

:pushpin: Differential rotation will apply automatically if `EVENT_TIME` differs from `REFERENCE_TIME`  
```IDL
IF (REFERENCE_TIME NE EVENT_TIME) THEN BEGIN
  ; Load smapbz from sav file
  restore, SAVE_ALL_DIR + '/' + REFERENCE_TIME + '/bxyz_submap.sav',/ver
  mapbz = drot_map(mapbz, time=smapbz.time)
```

#### Building map projection  
In IDL command line run  
`proj, IN_DIR`  
Where IN_DIR is `SAVE_ALL_DIR/REFERENCE_TIME`.  

If projected map doesn't satisfy, you could go to the first step, to re-select the Region-of-interest. You
also should :warning::warning::warning: delete the file `cmd_lat.sav` inside `IN_DIR`.  

:question: ***Should I use cmd_lat.sav from the REFERENCE (initial) TIME moment, picked from the beginning of analysis, for other time moments?***  _Because, for some moments in time, a size of data may be different. For example, at 9AM [576, 480] at 10AM with newly evaluated , but same coords range, may become [577, 480]. What could be the reason for such behaviour? Should I recalculate the center of the sun every hour?_  

#### Choosing a region for analysis with respect to projection effects  
In IDL command line:  
`plot_submap_m, IN_DIR`  
IN_DIR is picked from previous step.  
:pushpin: ROI selection  
```IDL
restore,IN_DIR+'/mapbxyz_m.sav',/ver
xrange = [-244., 44]
yrange = [-395., -155.5]
res = [0.5,0.5]
```
:question: ***In some future moments in time, since REFERENCE (initial) TIME, a size of data may be different. For example, at 9AM [\*576\*, 480] at 10AM with newly evaluated , but same coords range, may become [\*577\*, 480]. What could be the reason for such behaviour? Should I recalculate the center of the sun every hour? Not only rely on the reference time SOLAR_CENTER estimation?*** _Am I correct that I should keep the xrange and yrange the same (taking drot in account), for all the period of analysis [3 days]? But I also need to control the SUN_CENTER on the step1 for every time in the period?_  

:pushpin::warning: The shape of output data should be a multiple of a good number like 16, 32,...  
```IDL
; CHECK HERE THE SHAPE OF DATA. IT SHOULD BE MULTIPLE OF A GOOD NUMBER * 4
; Select the RANGE in the begin of file.
help, smapbz.data
```  

#### Preprocessing data for a Force-Free extrapolation  
In IDL run 
`pro creb_lv3_, IN_DIR`  
This script will create a level3_data (smoothed over 4x4 kernel) which will be used for a simulation.  
:warning: This step may fail if data_size cannot be divided by 4.  

#### Preparing boundary conditions for extrapolation in MPI-AMRVAC  
In IDL run 
`read_boundary, IN_DIR, amrvac_file, amrvac_file_nlfff`  
This script prepares binaries for `lfff` and `nlfff` for reading in MPI-AMRVAC.  
:pushpin: Carefully set the parameters in the beginning of file:  
```IDL
;=============================================
; Parameter needed to be set for your own case
;=============================================
nx = 144                ; x-size of the preprocessed magnetic field
ny = 120                ; y-size
arcsec2cm=7.29809e7; 2825e7     ; 1 arcsec in cm
xc=-100.23546*arcsec2cm ; x-coordinate of the center of the field of view
yc=-275.35374*arcsec2cm ; y-coordinate of the center
dx=4.0*0.50d*arcsec2cm  ; x spatial resolution of the preprocessed magnetic field
dy=dx                   ; y spatial resolution
;==============================================
```  
`xc`, `yc` could be found from the output of `help, smapbz` from previous step. `arcsec2cm` could be calculated from the known solar radius `6.9634e10 cm` & a radius in arcsec from the step 1: `(|x_l|+|x_r|)/2`. `nx`, `ny` is also the output shape of `help, smapbz` divided by 4.  

:information_source: This script will update the amrvac files, stored in `mmc/vmf/potential`, `mmc/vmf/nlfff` with x, y, z - ranges. You should **manually update** `nblocks` for parallelization in `.par` files.  

### Step 3. Data preparation pipeline for the whole timerange  
#### Automated boundary conditions creation with selected parameters from Step 2.  
Run in IDL  
`mmc/vmf/magnetic_preparation_pipeline.pro`  
:pushpin: Setup folders  
```IDL
; pro magnetic_preparation_pipeline
; ------ inputs --------------------------------
DATA_DIR = '/home/sunshine/repos/MagneticRoutines/downloads'
OUT_DIR = '/home/sunshine/data/event_20170906'
REFERENCE_TIME = '20170903_090000_TAI'

AMRVAC_FILE_POT = '/home/sunshine/repos/MagneticRoutines/mmc/vmf/potential/amrvac.par'
AMRVAC_FILE_NLFFF = '/home/sunshine/repos/MagneticRoutines/mmc/vmf/nlfff/amrvac.par'
; ------ end inputs ----------------------------

```
`DATA_DIR` is picked from python script from the Step 1.  

:x: ***Pipeline fails at 9:48, 4th timestep. Because DATA becomes of shape [577, 480]. This issue is related to previous questions***  

#### DAVE4VM velocity calculation for each timestep  
Run in IDL `mmc/idl_dave4vm/velocity_preparation.pro`  
:pushpin: Setup directories  
```IDL
; pro velocity_preparation
; ------ inputs --------------------------------
DATA_DIR = '/home/sunshine/repos/MagneticRoutines/downloads'
OUT_DIR = '/home/sunshine/data/event_20170906'
;;;;;; !!!!!!!!!!!! SET WINDOWSIZE
windowsize = 25
; ------ end inputs ----------------------------
```

The plasma velocity will be stored in `OUT_DIR/<EVENT_TIME>/level3_data/velocity_boundary.dat`  

:question: ***Am I correct that I use a magnetic field from `/level3_data/allboundaries.dat` files for velocity estimation with a DAVE4VM algorithm?***  

:question: ***I observe `NaNs` near the borders of calculated region, which are proportional to the windowsize. Should I increase the size of Region-of-Interest? Currently I substitute `NaNs` with `0`s. See image below.***    

![Result of DAVE4VM application to data](assets/images/svz_m.png)  

:question: ***Should I convert `velocity_boundary.dat` to a binary file for MPI-AMRVAC?***    

:white_check_mark: Result of data preparation pipeline could be found here: `assets/data`.  


### Step 4. Potential & NLFFF Extrapolation in MPI-AMRVAC  
In every Potential subdir like `20170903_090000_TAI/level3_data/extrapolation/potential` setup MPI-AMRVAC:  
```bash
setup.pl -d=3 -arch=default  
make -j 16
mpirun -np 192 ./amrvac -i amrvac.par
```  
Then in every subdir for NLFFF with Magnetofriction method, run the same sequence.  
:warning: If having issues run
```bash
make clean
```
If nblocks numbers are not good, update them as well.  

## :sun_with_face::question::sun_with_face:Further steps:sun_with_face::question::sun_with_face:  

:white_check_mark: Generally, I've completed preparation and downloading of data. I can reconstruct the B-field for every moment in time with MPI-AMRVAC (potential field requires seconds, nlfff requires ~40 minutes). I also can obtain a velocity of plasma on the bottom boundary with an adaptation of DAVE4VM. :warning: Some issues for a long time range analysis remain (see above). I believe that they are related a careful RoI choice & Sun center corrections.  

:question: ***Which next steps should I do to reproduce your paper [Data-driven Modeling of a Coronal Magnetic Flux Rope: From Birth to Death](https://doi.org/10.3847/1538-4357/ad088d)?***

:envelope: [shainalexander@yandex.ru](shainalexander@yandex.ru)  
  
Best Regards, Aleksandr (Alex) Shain  
