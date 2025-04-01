pro download_serial_hmi_by_list, size_pix = size_pix, dx_km = dx_km, out_dir = out_dir, tmp_dir = tmp_dir
  compile_opt idl2

  ; Step 1: Open an interactive window to select and load the file
  file = dialog_pickfile(/read, title = 'Select CSV File')
  if file eq '' then RETURN ; Exit if no file is selected

  ; Step 2: Read all lines from the file
  openr, unit, file, /get_lun ; Open the file for reading
  lines = list() ; Use a list to dynamically store lines
  while ~eof(unit) do begin
    line = ''
    readf, unit, line ; Read one line at a time
    if line ne '' then lines.add, line ; Append non-empty lines to the list
  endwhile
  free_lun, unit ; Close the file

  N = lines.count() ; Get the total number of lines

  ; Check if the file is empty or contains no valid data
  if N eq 0 then begin
    print, 'The file is empty or contains no valid data.'
    RETURN
  endif

  ; Initialize arrays
  observation_times = strarr(N)
  car_lonlat = fltarr(N, 2)
  c_xy = fltarr(N, 2)

  ; Step 3: Parse each line and populate the arrays
  for i = 1, N - 1 do begin
    ; Split the line into components using ';' as the delimiter
    parts = strsplit(lines[i], ',', /extract)
    if n_elements(parts) lt 3 then continue ; Skip invalid lines
    observation_times[i-1] = parts[0] ; Store the time as a string
    car_lonlat[i - 1, 0] = float(parts[1]) ; Store c_x as a float
    car_lonlat[i - 1, 1] = float(parts[2]) ; Store c_y as a float
    c_xy[i - 1, 0] = float(parts[3])
    c_xy[i - 1, 1] = float(parts[4])
  endfor

  ; CREATE DIFF DIR
  ;
  ;
  out_dir_diff = strjoin([out_dir, '_diff'], '')
  file_mkdir, out_dir_diff

  ; Step 4: Iterate over the arrays and print the required information
  for i = 0, N - 2 do begin
    ; PRINT, 'Time: ', observation_times[i], ' c_x: ', car_lonlat[i, 0], ' c_y: ', car_lonlat[i, 1]
    carrington_lon_lat = [car_lonlat[i, 0], car_lonlat[i, 1]]
    cur_time = observation_times[i]
    print, 'EVENT', i + 1, ' of', N, '. Time: ', cur_time, '. CAR LONLAT:', carrington_lon_lat
    gx_box_prepare_box, cur_time, carrington_lon_lat, size_pix, dx_km, out_dir = out_dir, tmp_dir = tmp_dir, /cea, /aia_euv, /carrington, /sfq
    if (i lt (N - 3)) then begin
      cur_xy = [c_xy[i, 0], c_xy[i, 1]]
      next_time = observation_times[i + 1]
      gx_box_prepare_box, next_time, cur_xy, size_pix, dx_km, out_dir = out_dir_diff, tmp_dir = tmp_dir, /cea, /sfq
    endif
  endfor
end
