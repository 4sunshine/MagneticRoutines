FUNCTION coord_at_time, initial_xy = initial_xy, initial_time = initial_time, target_time = target_time

target_coord = FINDGEN(SIZE(target_time, /N_ELEMENTS), 2)

FOREACH t, target_time, index DO BEGIN
  daylong = anytim('2017-09-03 00:00:00')-anytim('2017-09-02 00:00:00')
  ll=arcmin2hel(initial_xy(0)/60., initial_xy(1)/60., date = anytim(t, /yohkoh, /date))
  lat = ll(0)
  lon = ll(1)
  lonn = lon + diff_rot((anytim(t) - anytim(initial_time)) / daylong, lat)
  new_coord = ROUND(hel2arcmin(lat, lonn, date = anytim(t, /yohkoh, /date))*60.)
  target_coord[index, *] = new_coord
ENDFOREACH

RETURN, target_coord
END

FUNCTION make_boxes, times = times, centres = centres, size_pix = size_pix, dx_km = dx_km, out_dir = out_dir, tmp_dir = tmp_dir
  FOREACH t, times, index DO BEGIN
    gx_box_prepare_box, t, centres[index], size_pix, dx_km, out_dir = out_dir, /cea, tmp_dir = tmp_dir, /aia_euv
  ENDFOREACH
RETURN, 1
END