function read_allboundaries, _data_dir
  ; Reads allboundaries.dat file
  
  vector_sav = _data_dir + '/smapbxyz_m.sav'
  restore, vector_sav, /ver
  _time = smapbz.time
  destroy, smapbx
  destroy, smapby
  destroy, smapbz

  data_dir = _data_dir + '/level3_data'

  grid_file = data_dir + '/grid.ini'

  get_lun,u
  openr,u, grid_file
  _ = ''
  readf,u,_
  readf,u,nx
  readf,u,_
  readf,u,ny
  readf,u,_
  readf,u,nz
  close,u
  free_lun,u
  
  nx = FIX(strtrim(nx,2))
  ny = FIX(strtrim(ny,2))

  bx = dblarr(nx, ny)
  by = dblarr(nx, ny)
  bz = dblarr(nx, ny)

  boundary_file = data_dir + '/allboundaries.dat'
  openr, u, boundary_file
  for iy=0,ny-1 do begin
    for ix=0,nx-1 do begin
      _cur_val = ''
      readf,u, _cur_val
      bx[ix,iy] = DOUBLE(_cur_val)
      readf,u, _cur_val
      by[ix,iy] = DOUBLE(_cur_val)
      readf,u, _cur_val
      bz[ix,iy] = DOUBLE(_cur_val)
    endfor
  endfor
  close,u
  free_lun,u
  return, {name: '', BXB: bx, BYB: by, BZB: bz, TIME: _time}
end

function read_dxdy, data_dir
  ; Reads dxdy in cm
  dxdy_file = data_dir + '/level3_data/dxdy_cm.txt'

  get_lun,u
  openr,u, dxdy_file
  _ = ''
  dxdy = ''
  readf,u,_
  readf,u,dxdy

  close,u
  free_lun,u

  words = STRSPLIT(dxdy, /EXTRACT)
  dx = DOUBLE(words[0])
  dy = DOUBLE(words[1])

  return, {name: '', DX: dx, DY: dy, UNIT: 'cm'}
end


function replace_nans, array
  ; Replaces NaNs with zeros
  bad = WHERE(~FINITE(array), count)
  IF count GT 0 THEN array[bad] = 0.
  return, array
end


function scaled_map, refmap, data, id
  ; Prepares a map for 4-times downscaled data 
  SCALE = 4.d0
  x = {$
    DATA: data, XC: refmap.XC, YC: refmap.YC,$
    DX: SCALE * refmap.DX, DY: SCALE * refmap.DY,$
    TIME: refmap.TIME, DUR: refmap.DUR, UNITS: refmap.UNITS,$
    ID: id, ROLL_ANGLE: refmap.ROLL_ANGLE, ROLL_CENTER: refmap.ROLL_CENTER $
  }
  return, x
end

pro write_2d_array_dat, dat_path, ax, ay, az
  ; Writes array as readable .dat file
  nd = SIZE(ax, /DIMENSION)
  nx = nd[0]
  ny = nd[1]

  get_lun, u
  openw, u, dat_path
  for iy=0,ny-1 do begin
    for ix=0,nx-1 do begin
      printf,u, ax[ix,iy]
      printf,u, ay[ix,iy]
      printf,u, az[ix,iy]
    endfor
  endfor
  close,u
  free_lun,u

end

; pro velocity_preparation
; ------ inputs --------------------------------
DATA_DIR = '/home/sunshine/repos/MagneticRoutines/downloads'
OUT_DIR = '/home/sunshine/data/event_20170906'

; ------ end inputs ----------------------------

timestamps = DATA_DIR + '/times_paired.txt'

get_lun, lun_in
openr, lun_in, timestamps

while (~EOF(lun_in)) do begin
  cur_pair = ''
  ; Read one line at a time
  readf, lun_in, cur_pair
  words = STRSPLIT(cur_pair, /EXTRACT)
  time_start = words[0]
  time_end = words[1]
  
  CUR_OUT_DIR = OUT_DIR + '/' + time_start
  start_data_dir = OUT_DIR + '/' + time_start ;+ '/level3_data'
  dxdy = read_dxdy(start_data_dir)

  b_start = read_allboundaries(start_data_dir)
  end_data_dir = OUT_DIR + '/' + time_end ;+ '/level3_data'
  b_stop = read_allboundaries(end_data_dir)
  start_seconds = utc2sec(anytim2utc(b_start.TIME))
  end_seconds = utc2sec(anytim2utc(b_stop.TIME))
  dt = double(end_seconds - start_seconds)

  DX = double(dxdy.DX / 1.d5) ; km
  DY = double(dxdy.DY / 1.d5) ; km
  
  ;;;;;; !!!!!!!!!!!! SET WINDOWSIZE
  windowsize = 25
  
  bx_start = b_start.BXB
  by_start = b_start.BYB
  bz_start = b_start.BZB
  
  bx_stop = b_stop.BXB
  by_stop = b_stop.BYB
  bz_stop = b_stop.BZB
  
  destroy, b_start
  destroy, b_stop

  do_dave4vm_and, bx_start, by_start, bz_start, bx_stop, by_stop, bz_stop,$
    DX, DY, DT, vel4vm, magvm, windowsize = windowsize
    
  ; print, vel4vm.U0 ; V0, W0 -- uvw == xyz

  vz = vel4vm.W0
  vy = vel4vm.V0
  vx = vel4vm.U0

  vx = replace_nans(vx)
  vy = replace_nans(vy)
  vz = replace_nans(vz)

  velocity_dat_file = CUR_OUT_DIR + '/level3_data/velocity_boundary.dat'

  write_2d_array_dat, velocity_dat_file, vx, vy, vz

  vector_sav = CUR_OUT_DIR + '/smapbxyz_m.sav'
  restore, vector_sav, /ver

  x = scaled_map(smapbz, vz, "Vz")

  wdef,1,1300,800
  !p.background=255
  plot_map,x,color=0,bcolor=0,charsize=2.0,dmax=1.5,dmin=-1.5
  !p.background=0
  write_png, CUR_OUT_DIR+'/svz_m.png', tvrd()

endwhile

free_lun, lun_in

END
