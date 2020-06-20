;start time
tstart = '2017-09-03 22:00:00'
;end_time
tend = '2017-09-04 09:11:00'
;step_time
tstep = '00:12:00'
;initial_conditions
centre = [15.,-270.]; at 22:00
out_dir = 'C:\AppsData\gx_out'
tmp_dir = 'C:\AppsData\gx_temp'
dx_km=400.
size_pix=[400,400,75]

download_serial_hmi, tstart = tstart, tend = tend, tstep = tstep, centre = centre, $
                      out_dir = out_dir, tmp_dir = tmp_dir, dx_km = dx_km, size_pix = size_pix
END