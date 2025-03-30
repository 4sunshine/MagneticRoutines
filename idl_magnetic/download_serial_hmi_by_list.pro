PRO download_serial_hmi_by_list, size_pix = size_pix, dx_km = dx_km, out_dir = out_dir, tmp_dir = tmp_dir

    ; Step 1: Open an interactive window to select and load the file
    file = DIALOG_PICKFILE(/READ, TITLE='Select CSV File')
    IF file EQ '' THEN RETURN  ; Exit if no file is selected

    ; Step 2: Read all lines from the file
    OPENR, unit, file, /GET_LUN  ; Open the file for reading
    lines = LIST()  ; Use a list to dynamically store lines
    WHILE ~EOF(unit) DO BEGIN
        line = ''
        READF, unit, line  ; Read one line at a time
        IF line NE '' THEN lines.ADD, line  ; Append non-empty lines to the list
    ENDWHILE
    FREE_LUN, unit  ; Close the file

    N = lines.COUNT()  ; Get the total number of lines

    ; Check if the file is empty or contains no valid data
    IF N EQ 0 THEN BEGIN
        PRINT, 'The file is empty or contains no valid data.'
        RETURN
    ENDIF

    ; Initialize arrays
    observation_times = STRARR(N)
    c_xy = FLTARR(N, 2)

    ; Step 3: Parse each line and populate the arrays
    FOR i = 0, N-1 DO BEGIN
        ; Split the line into components using ';' as the delimiter
        parts = STRSPLIT(lines[i], ',', /EXTRACT)
        IF N_ELEMENTS(parts) LT 3 THEN CONTINUE  ; Skip invalid lines
        observation_times[i] = parts[0]  ; Store the time as a string
        c_xy[i, 0] = FLOAT(parts[1])    ; Store c_x as a float
        c_xy[i, 1] = FLOAT(parts[2])    ; Store c_y as a float
    ENDFOR

    ; Step 4: Iterate over the arrays and print the required information
    FOR i = 0, N-1 DO BEGIN
        ; PRINT, 'Time: ', observation_times[i], ' c_x: ', c_xy[i, 0], ' c_y: ', c_xy[i, 1]
        carrington_lon_lat = [c_xy[i, 0], c_xy[i, 1]]
        cur_time = observation_times[i]
        PRINT, 'EVENT', i+1, ' of', N, '. Time: ', cur_time, '. CAR LONLAT:', carrington_lon_lat
        gx_box_prepare_box, cur_time, carrington_lon_lat, size_pix, dx_km, out_dir = out_dir, tmp_dir = tmp_dir, /cea, /aia_euv, /carrington
    ENDFOR

END