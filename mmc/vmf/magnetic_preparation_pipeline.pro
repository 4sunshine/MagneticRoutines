; pro magnetic_preparation_pipeline
; ------ inputs --------------------------------
DATA_DIR = '/home/sunshine/repos/MagneticRoutines/downloads'
OUT_DIR = '/home/sunshine/data/event_20170906'
REFERENCE_TIME = '20170903_090000_TAI'

AMRVAC_FILE_POT = '/home/sunshine/repos/MagneticRoutines/mmc/vmf/potential/amrvac.par'
AMRVAC_FILE_NLFFF = '/home/sunshine/repos/MagneticRoutines/mmc/vmf/nlfff/amrvac.par'
; ------ end inputs ----------------------------

timestamps = DATA_DIR + '/timestamps.txt'

get_lun, lun_in
openr, lun_in, timestamps

while (~EOF(lun_in)) do begin
  cur_time = ''
  ; Read one line at a time
  readf, lun_in, cur_time
  ; words = STRSPLIT(x, /EXTRACT)
  if FILE_TEST(OUT_DIR + '/' + cur_time, /DIRECTORY) then continue
  prepare_hmi, DATA_DIR, cur_time, REFERENCE_TIME, OUT_DIR
  cur_dir = OUT_DIR + '/' + cur_time
  proj, cur_dir
  plot_submap_m, cur_dir
  creb_lv3_, cur_dir
  read_boundary, cur_dir, amrvac_file_pot, amrvac_file_nlfff

endwhile

free_lun, lun_in

END
