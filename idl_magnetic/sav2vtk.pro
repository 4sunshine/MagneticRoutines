PRO sav2vtk
  files = DIALOG_PICKFILE(TITLE = 'MAGNETIC BOXES SELECT', /MULTIPLE_FILES)
  sz = size(files)

  FOR m=0, sz[1]-1 DO BEGIN

    restore, files[m]
    print,'task',m+1,' of',sz[1]
    dim=size(box.bx)

    nx=dim(1)
    ny=dim(2)
    nz=dim(3)

    ; Writing a vector field 

    openw,lun,FILE_BASENAME(files[m], '.sav') + '.vtk',/get_lun
    
    printf,lun,'# vtk DataFile Version 2.0'
    printf,lun,'Vector magnetic field B';'Curl of magnetic field';'Vector magnetic field b'
    printf,lun,'ASCII'
    printf,lun,'DATASET STRUCTURED_POINTS'
    printf,lun,'DIMENSIONS',nx,ny,nz
    printf,lun,'ORIGIN', 0.000, 0.000, 0.000
    printf,lun,'SPACING', 1.000, 1.000, 1.000
    printf,lun,'POINT_DATA', nx*ny*nz
    printf,lun,'VECTORS B float';vector
    for k=0, nz-1 do begin
      for j=0, ny-1 do begin
        for i=0, nx-1 do begin
          printf,lun,box.bx(i,j,k),box.by(i,j,k),box.bz(i,j,k)
        endfor
      endfor
    endfor
    free_lun,lun
    
    ; End writing a vector field
    
    ; Calculating curl (rot) of a vector field
    
    curl, box.bx, box.by, box.bz, cx, cy, cz
    
    ; Writing curl of a vector field

    openw,lun,FILE_BASENAME(files[m], '.sav') + '_rot.vtk',/get_lun

    printf,lun,'# vtk DataFile Version 2.0'
    printf,lun,'Curl of vector magnetic field B'
    printf,lun,'ASCII'
    printf,lun,'DATASET STRUCTURED_POINTS'
    printf,lun,'DIMENSIONS',nx,ny,nz
    printf,lun,'ORIGIN', 0.000, 0.000, 0.000
    printf,lun,'SPACING', 1.000, 1.000, 1.000
    printf,lun,'POINT_DATA', nx*ny*nz
    printf,lun,'VECTORS rotB float';vector
    for k=0, nz-1 do begin
      for j=0, ny-1 do begin
        for i=0, nx-1 do begin
          printf,lun,cx(i,j,k),cy(i,j,k),cz(i,j,k)
        endfor
      endfor
    endfor
    free_lun,lun
    
    ; End writing curl of a vector field
    
    undefine, box
  ENDFOR
end
