;pro read_boundary
IN_DIR='~/data/20170903_090000_TAI'

amrvac_file = '~/repos/magnetic_modeling_codes/example/vmf_analysis/03preprocess/potential/amrvac.par'
amrvac_file_nlfff = '~/repos/magnetic_modeling_codes/example/vmf_analysis/03preprocess/nlfff/amrvac.par'
; Paste from output of help, smapbz

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

x = 0.0
bx=fltarr(nx,ny)
by=fltarr(nx,ny)
bz=fltarr(nx,ny)
get_lun,u
openr,u,IN_DIR+'/level3_data/allboundaries.dat'
for j=0,ny-1 do begin
  for i=0,nx-1 do begin
    readf,u,x
    bx[i,j]=x
    readf,u,x
    by[i,j]=x
    readf,u,x
    bz[i,j]=x
  endfor
endfor
close,u
free_lun,u
tvscl,bz
;v=vector(bx,by)

; Used when additional "ghost" pixels are added
; tmp = Bz[2:nx-3,2:ny-3]
;Bz = tmp
sizebz=size(Bz)
nx1=sizebz[1]
nx2=sizebz[2]
out_dir=IN_DIR+'/level3_data/extrapolation/potential/potential_boundary'
amrvac_out_dir = IN_DIR+'/level3_data/extrapolation/potential
if ~file_test(out_dir, /directory) then file_mkdir, out_dir
data_out_dir = amrvac_out_dir + '/data'
if ~file_test(data_out_dir, /directory) then file_mkdir, data_out_dir
filename=out_dir + '/potential_boundary.dat'
openw,lun,filename,/get_lun
writeu,lun,nx1
writeu,lun,nx2
writeu,lun,double(xc)
writeu,lun,double(yc)
writeu,lun,double(dx)
writeu,lun,double(dy)
writeu,lun,double(Bz)
free_lun,lun
print,'Bz range (Gauss):', min(Bz),max(Bz)
;wdef,2,800,800
;tvscl,Bz
print,'Computation domain for potential field:'
print,'nx1,nx2',nx1,nx2
print,'xc,yc (cm)',xc,yc
print,'dx,dy (cm)',dx,dy
x1=xc-nx1*dx/2
x2=xc+nx1*dx/2
y1=yc-nx2*dy/2
y2=yc+nx2*dy/2

fmt_xmin1 = '        xprobmin1='+strtrim(string(x1*1.e-9),2)+'d0'
fmt_xmax1 = '        xprobmax1='+strtrim(string(x2*1.e-9),2)+'d0'
fmt_xmin2 = '        xprobmin2='+strtrim(string(y1*1.e-9),2)+'d0'
fmt_xmax2 = '        xprobmax2='+strtrim(string(y2*1.e-9),2)+'d0'
fmt_xmin3 = '        xprobmin3='+strtrim(string(0.0*1.e-9+0.1),2)+'d0'   ; to lift the domain 1 Mm above 0
fmt_xmax3 = '        xprobmax3='+strtrim(string((y2-y1)*1.e-9),2)+'d0'

; output 
print,'x,y, and z range (10 Mm):'
print,fmt_xmin1
print,fmt_xmax1
print,fmt_xmin2
print,fmt_xmax2
print,fmt_xmin3
print,fmt_xmax3

; ASHAIN: write the same to a file
; 
; 
; Open the file for writing
openw, lun, out_dir + '/meta_amrvac.txt', /get_lun

; Write each line to the file
printf, lun, 'x,y, and z range (10 Mm):'
printf, lun, fmt_xmin1
printf, lun, fmt_xmax1
printf, lun, fmt_xmin2
printf, lun, fmt_xmax2
printf, lun, fmt_xmin3
printf, lun, fmt_xmax3

; Close the file and free the logical unit number
free_lun, lun

extract_and_update_xprob, amrvac_out_dir, amrvac_file, fmt_xmin1, fmt_xmin2, fmt_xmin3, fmt_xmax1, fmt_xmax2, fmt_xmax3, nx, ny


; MAKE FOR NLFFF
; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Should pad 2 zeros each side

x = 0.0
bx=fltarr(nx,ny)
by=fltarr(nx,ny)
bz=fltarr(nx,ny)
get_lun,u
openr,u,IN_DIR+'/level3_data/allboundaries.dat'
for j=0,ny-1 do begin
  for i=0,nx-1 do begin
    readf,u,x
    bx[i,j]=x
    readf,u,x
    by[i,j]=x
    readf,u,x
    bz[i,j]=x
  endfor
endfor
close,u
free_lun,u
tvscl,bz
;v=vector(bx,by)

bx_t=fltarr(nx+4, ny+4)
by_t=fltarr(nx+4, ny+4)
bz_t=fltarr(nx+4, ny+4)

bx_t[2: nx+1, 2: ny+1] = bx
by_t[2: nx+1, 2: ny+1] = by
bz_t[2: nx+1, 2: ny+1] = bz

bx = bx_t
by = by_t
bz = bz_t

sizebz=size(Bz)
nx1=sizebz[1]
nx2=sizebz[2]
out_dir=IN_DIR+'/level3_data/extrapolation/nlfff_boundary'
amrvac_out_dir = IN_DIR+'/level3_data/extrapolation/
if ~file_test(out_dir, /directory) then file_mkdir, out_dir
out_dir_data = IN_DIR+'/level3_data/extrapolation/data'
if ~file_test(out_dir_data, /directory) then file_mkdir, out_dir_data
filename=out_dir + '/nlfff_boundary.dat'

openw,lun,filename,/get_lun
writeu,lun,nx1
writeu,lun,nx2
writeu,lun,double(xc)
writeu,lun,double(yc)
writeu,lun,double(dx)
writeu,lun,double(dy)
writeu,lun,double(Bx)
writeu,lun,double(By)
writeu,lun,double(Bz)
free_lun,lun
print,'Bz range (Gauss):', min(Bz),max(Bz)
;wdef,2,800,800
;tvscl,Bz
print,'Computation domain for nonlinear force-free field:'
nx1=nx1-4
nx2=nx2-4
print,'nx1,nx2',nx1,nx2
print,'xc,yc (cm)',xc,yc
print,'dx,dy (cm)',dx,dy
x1=xc-nx1*dx/2
x2=xc+nx1*dx/2
y1=yc-nx2*dy/2
y2=yc+nx2*dy/2
; output
fmt_xmin1 = '        xprobmin1='+strtrim(string(x1*1.e-9),2)+'d0'
fmt_xmax1 = '        xprobmax1='+strtrim(string(x2*1.e-9),2)+'d0'
fmt_xmin2 = '        xprobmin2='+strtrim(string(y1*1.e-9),2)+'d0'
fmt_xmax2 = '        xprobmax2='+strtrim(string(y2*1.e-9),2)+'d0'
fmt_xmin3 = '        xprobmin3='+strtrim(string(0.0*1.e-9+0.1),2)+'d0'   ; to lift the domain 1 Mm above 0
fmt_xmax3 = '        xprobmax3='+strtrim(string((y2-y1)*1.e-9),2)+'d0'

; output 
print,'x,y, and z range (10 Mm):'
print,fmt_xmin1
print,fmt_xmax1
print,fmt_xmin2
print,fmt_xmax2
print,fmt_xmin3
print,fmt_xmax3

; ASHAIN: write the same to a file
;
;
; Open the file for writing
openw, lun, out_dir + '/meta_amrvac.txt', /get_lun

; Write each line to the file
printf, lun, 'x,y, and z range (10 Mm):'
printf, lun, fmt_xmin1
printf, lun, fmt_xmax1
printf, lun, fmt_xmin2
printf, lun, fmt_xmax2
printf, lun, fmt_xmin3
printf, lun, fmt_xmax3

; Close the file and free the logical unit number
free_lun, lun

extract_and_update_xprob, amrvac_out_dir, amrvac_file_nlfff, fmt_xmin1, fmt_xmin2, fmt_xmin3, fmt_xmax1, fmt_xmax2, fmt_xmax3, nx1, nx2


end
