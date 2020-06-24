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

    openw,lun,FILE_BASENAME(files[m], '.sav') + '.vtk',/get_lun
    
    printf,lun,'# vtk DataFile Version 2.0'
    printf,lun,'Vector magnetic field B';'Curl of magnetic field';'Vector magnetic field b'
    printf,lun,'ASCII'
    printf,lun,'DATASET STRUCTURED_POINTS'
    printf,lun,'DIMENSIONS',nx,ny,nz
    printf,lun,'ORIGIN', 0.000, 0.000, 0.000
    printf,lun,'SPACING', 1.000, 1.000, 1.000
    printf,lun,'POINT_DATA', nx*ny*nz
    printf,lun,'VECTORS Bnlfffe float';vector
    for k=0, nz-1 do begin
      for j=0, ny-1 do begin
        for i=0, nx-1 do begin
          printf,lun,box.bx(i,j,k),box.by(i,j,k),box.bz(i,j,k)
        endfor
      endfor
    endfor
    free_lun,lun
    undefine, box
  ENDFOR
end




