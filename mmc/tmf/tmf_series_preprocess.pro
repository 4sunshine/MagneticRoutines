; pro tmf_series_preprocess

; ------ inputs --------------------------------
DATA_DIR = '/home/sunshine/repos/MagneticRoutines/downloads'
OUT_DIR = '/home/sunshine/data/event_20170906'
REFERENCE_TIME = '20170903_090000_TAI'

AMRVAC_FILES_DIR = '/home/sunshine/repos/MagneticRoutines/mmc/tmf/tmf_par'

NUM_GHOST_CELLS = 2  ;; Num Ghost Cells Per Side
; ------ end inputs ----------------------------


timestamps = DATA_DIR + '/timestamps.txt'
NT_REF = count_nonempty_lines(timestamps)  ; num timestamps for analysis

if NT_REF lt 2 then return

nx_ny_reference_file = OUT_DIR + '/' + REFERENCE_TIME + '/level3_data/grid.ini'
nx_ny_ref = read_grid(nx_ny_reference_file)
NX_REF = nx_ny_ref.NX + 2 * NUM_GHOST_CELLS
NY_REF = nx_ny_ref.NY + 2 * NUM_GHOST_CELLS

print, "NX_REF:" + string(NX_REF) + " NY_REF:" + string(NY_REF) + " NT_REF:" + string(NT_REF)

Bvec = fltarr(NX_REF, NY_REF, NT_REF, 3)
Vvec = fltarr(NX_REF, NY_REF, NT_REF, 3)
Tarr = fltarr(NT_REF)

; Creating directories
TMF_OUT_DIR = OUT_DIR + '/tmf'
if ~file_test(TMF_OUT_DIR, /directory) then file_mkdir, TMF_OUT_DIR
TMF_OUT_DATA_DIR = TMF_OUT_DIR + '/data'
if ~file_test(TMF_OUT_DATA_DIR, /directory) then file_mkdir, TMF_OUT_DATA_DIR

;;; Main Loop

get_lun, lun_in
openr, lun_in, timestamps

cur_time_id = 0
prev_time_val = 0
t_0 = 0

while (~EOF(lun_in)) do begin
  cur_time = ''
  ; Read one line at a time
  readf, lun_in, cur_time
  cur_dir = OUT_DIR + '/' + cur_time
  print, "Current Time:", cur_time
  data = read_vb_boundary(cur_dir, num_ghost_cells)

  if data.NX ne NX_REF then begin
    message, "Error: data.NX != NX_REF", data.NX
    return
  endif

  if data.NY ne NY_REF then begin
    message, "Error: data.NY != NY_REF", data.NY
    return
  endif

  Bvec[*, *, cur_time_id, 0] = data.BX ;; data.BX is 2D-Array of (nx, ny) shape
  Bvec[*, *, cur_time_id, 1] = data.BY
  Bvec[*, *, cur_time_id, 2] = data.BZ

  Vvec[*, *, cur_time_id, 0] = data.VX
  Vvec[*, *, cur_time_id, 1] = data.VY
  Vvec[*, *, cur_time_id, 2] = data.VZ

  if cur_time_id eq 0 then init_time_val = data.T
  cur_time_dt = data.T - init_time_val
  Tarr[cur_time_id] = cur_time_dt

  ; Copy files for potential field extrapolation from initial step
  if cur_time_id eq 0 then begin
    potential_data_dir = cur_dir + '/level3_data/extrapolation/potential'
    print, "BE CAREFUL, COPYING FILES FOR POTENTIAL FIELD FROM !!!", potential_data_dir

    response = dialog_message("Copy Potential Files? (Y/N)", /question, /cancel)
    if strupcase(response) eq 'YES' then begin
      print, "Copying ..."
      spawn,'cp -r ' + potential_data_dir + ' ' + TMF_OUT_DIR
    endif else begin
      print, "Potential field files were not copied"
    endelse

  endif

  cur_time_id = cur_time_id + 1

endwhile

close, lun_in
free_lun, lun_in

bin_write_b_data, TMF_OUT_DIR, 'B_boundary.dat', Bvec*1.0d0, Tarr*1.0d0,$
  NX_REF, NY_REF, NT_REF, data.DX, data.DY
  
bin_write_b_data, TMF_OUT_DIR, 'V_boundary.dat', Vvec*1.0d0, Tarr*1.0d0,$
  NX_REF, NY_REF, NT_REF, data.DX, data.DY

print, "BE SURE TO MODIFY AMRVAC FILES from ", TMF_OUT_DIR
spawn,'cp ' + AMRVAC_FILES_DIR + '/* ' + TMF_OUT_DIR

END
