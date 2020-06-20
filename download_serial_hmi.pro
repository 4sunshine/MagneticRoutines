PRO download_serial_hmi, tstart = tstart, tend = tend, tstep = tstep, centre = centre, $
                              size_pix = size_pix, dx_km = dx_km, out_dir = out_dir, tmp_dir = tmp_dir
;
; THIS IS ROUTINE TO DOWNLOAD HMI AND AIA DATA IN SERIAL MANNER AND CREATE POTENTIAL FIELD BOXES
; 
; 1 arcsec = 727 km
; HMI resolution = 1 arcsec, Pixel size = 1/2 arcsec
;initial_conditions

;computing_count_of_tasks

daylong = anytim('2017-09-03 00:00:00')-anytim('2017-09-02 00:00:00')
cot = ROUND((anytim(tend)-anytim(tstart))/anytim(tstep))
print,'count_of_tasks =',cot
anytstep = anytim(tstep)

;serial_work
FOR i = 0, cot DO BEGIN
  print,'task',i+1,' of ',cot+1
;  print, centre
  itime = anytim(tstart)+i*anytstep
  print,anytim(itime,/yohkoh)
  gx_box_prepare_box, itime, centre, size_pix, dx_km, out_dir = out_dir,/cea, tmp_dir = tmp_dir, /aia_euv
  ll=arcmin2hel(centre(0)/60.,centre(1)/60.,date = anytim(itime,/yohkoh,/date))
  lat = ll(0)
  lon = ll(1)
  lonn = lon + diff_rot(anytstep/daylong, lat)
  centre = ROUND(hel2arcmin(lat,lonn,date = anytim(itime,/yohkoh,/date))*60.)
ENDFOR

END