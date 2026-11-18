function read_boundary_dat, file, nx, ny

  if ~file_test(file) then begin
    message, 'ERROR: Boundary .dat file ' + file + ' does not exist';, /error
    return, -1
  endif

  fx=fltarr(nx,ny)
  fy=fltarr(nx,ny)
  fz=fltarr(nx,ny)

  openr,u,file,/get_lun
  for j=0,ny-1 do begin
    for i=0,nx-1 do begin
      readf,u,x
      fx[i,j]=x
      readf,u,x
      fy[i,j]=x
      readf,u,x
      fz[i,j]=x
    endfor
  endfor
  close,u
  free_lun,u

  return, {name: '', FX: fx, FY: fy, FZ: fz}
end

function read_grid, grid_file

  if ~file_test(grid_file) then begin
    message, 'ERROR: grid .ini file ' + grid_file + ' does not exist'; , /error
    return, -1
  endif

  openr,u, grid_file,/get_lun
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
  
  return, {name: '', NX: nx, NY: ny}
end

function add_ghost_cells, array, num_ghost_cells
  ; Pad (add ghost cells) for 2D arrays with zeros around each side
  ; Input:
  ;   array - 2D input array to be padded (e.g., [144, 120])
  ;   num_ghost_cells - number of ghost cells to add to each side
  ; Output:
  ;   upd_array - padded array with zeros
  
  ; Get original dimensions
  if num_ghost_cells eq 0 then return, array * 1.0   ; Forces a copy

  dims = size(array, /dimensions)
  nx = dims[0]
  ny = dims[1]
  
  ; Calculate new dimensions
  new_nx = nx + 2 * num_ghost_cells
  new_ny = ny + 2 * num_ghost_cells
  
  ; Create new array filled with zeros
  upd_array = fltarr(new_nx, new_ny)
  
  ; Insert original array in the center
  upd_array[num_ghost_cells:num_ghost_cells+nx-1, $
            num_ghost_cells:num_ghost_cells+ny-1] = array
  
  return, upd_array
end

function add_ghost_cells_to_field_struct, f, num_ghost_cells
  ; Pad (add ghost cells) for (FX, FY, FZ) struct of 2D arrays with zeros around each side

  fx_ = add_ghost_cells(f.FX, num_ghost_cells)
  fy_ = add_ghost_cells(f.FY, num_ghost_cells)
  fz_ = add_ghost_cells(f.FZ, num_ghost_cells)

  return, {name: '', FX: fx_, FY: fy_, FZ: fz_}
end

function suppress_v_in_weak_b, vx, vy, vz, bx, by, bz, threshold
  ; Sets vx, vy, vz to zero where |B| < threshold
  ; Input:
  ;   vx, vy, vz - velocity components (2D arrays)
  ;   bx, by, bz - magnetic field components (2D arrays)
  ;   threshold  - B magnitude threshold value
  ; Output:
  ;   structure with updated velocity components
  
  ; Check all arrays have same dimensions
  dims_vx = size(vx, /dimensions)
  dims_vy = size(vy, /dimensions)
  dims_vz = size(vz, /dimensions)
  dims_bx = size(bx, /dimensions)
  dims_by = size(by, /dimensions)
  dims_bz = size(bz, /dimensions)
  
  if not (array_equal(dims_vx, dims_vy) and array_equal(dims_vx, dims_vz) and $
          array_equal(dims_vx, dims_bx) and array_equal(dims_vx, dims_by) and $
          array_equal(dims_vx, dims_bz)) then begin
      message, 'All input arrays must have the same dimensions';, /error
      return, -1
  endif
  
  ; Calculate |B| magnitude
  b_mag = sqrt(bx^2 + by^2 + bz^2)
  
  ; Create mask where |B| < threshold
  weak_b_mask = (b_mag lt threshold)
  
  ; Apply mask to velocity components
  vx_upd = vx * (1 - weak_b_mask)
  vy_upd = vy * (1 - weak_b_mask)
  vz_upd = vz * (1 - weak_b_mask)
  
  return, {name: 'suppressed_velocity', FX: vx_upd, FY: vy_upd, FZ: vz_upd}
end

function read_text_line, filename
  ; Initialize output string
  line = ''
  ; Open the file for reading
  openr, lun, filename, /get_lun
  ; Read one line
  readf, lun, line
  ; Close the file
  close, lun
  ; Free the LUN
  free_lun, lun
  ; Strip leading/trailing whitespace and return
  return, strtrim(line, 2)
end


function read_vb_boundary, data_dir, num_ghost_cells

  dxdy = read_dxdy(data_dir)
  cur_data_dir = data_dir + '/level3_data'
  grid_file = cur_data_dir + '/grid.ini'
  timestamp_file = cur_data_dir + '/timestamp.txt'
  b_field_file = cur_data_dir + '/allboundaries.dat'
  v_field_file = cur_data_dir + '/velocity_boundary.dat'

  grid = read_grid(grid_file)
  Bf = read_boundary_dat(b_field_file, grid.NX, grid.NY)

  Vf_ = read_boundary_dat(v_field_file, grid.NX, grid.NY)
  Vf = suppress_v_in_weak_b(Vf_.FX, Vf_.FY, Vf_.FZ, Bf.FX, Bf.FY, Bf.FZ, 10.)

  B = add_ghost_cells_to_field_struct(Bf, num_ghost_cells)
  V = add_ghost_cells_to_field_struct(Vf, num_ghost_cells)
  t_ = read_text_line(timestamp_file)
  t_ = utc2sec(anytim2utc(t_))
  
  dx_ = dxdy.DX
  dy_ = dxdy.DY
  ;;; See mod_usr.t for time_driven_mf !!!
  if dxdy.UNIT eq 'cm' then begin
    ; convert to km
    dx_ = dx_ / 1.e5
    dy_ = dy_ / 1.e5
    dxdy.UNIT = 'km'
  endif
  
  nx_ = grid.NX + 2*num_ghost_cells
  ny_ = grid.NY + 2*num_ghost_cells

  return, {name: '',$
    BX: B.FX, BY: B.FY, BZ: B.FZ, B_UNIT: 'Gauss',$
    VX: V.FX, VY: V.FY, VZ: V.FZ, V_UNIT: 'km/s',$
    NX: nx_, NY: ny_,$
    T: t_, T_UNIT: 's',$
    DX: dx_, DY: dy_, DX_UNIT: dxdy.UNIT}

END

function count_nonempty_lines, filename
  ; Initialize counter
  count = 0

  ; Open the file for reading
  openr, lun, filename, /get_lun

  ; Loop until end of file
  while ~EOF(lun) do begin
    line = ''
    readf, lun, line
    
    ; Only count non-empty lines
    if STRTRIM(line, 2) ne '' then count++
  endwhile

  ; Clean up
  close, lun
  free_lun, lun
  
  ; Return result
  return, count
end


pro bin_write_b_data, out_dir, filename, Barr, Tarr, nx, ny, nt, dx, dy
  ; TODO: check ghost cells
  data_ibc_dir = out_dir + '/data_ibc'
  out_filename = data_ibc_dir + '/' + filename
  if ~file_test(data_ibc_dir, /directory) then file_mkdir, data_ibc_dir

  nx_dd = intarr(3)
  nx_dd[0] = nx
  nx_dd[1] = ny
  nx_dd[2] = nt

  dxa = fltarr(3)
  dxa[0] = dx
  dxa[1] = dy
  dxa[2] = dy ;; TODO: check it later
  dxa = double(dxa)

  bddb = double(Barr)
  t_driven_b = double(Tarr)

  openw,lun,out_filename,/get_lun
  writeu,lun,long(nx)
  writeu,lun,long(ny)
  writeu,lun,long(nt)
  writeu,lun,double(dx)
  writeu,lun,double(dy)
  writeu,lun,double(dy)
  writeu,lun,double(bddb)
  writeu,lun,double(t_driven_b)
  free_lun,lun

end