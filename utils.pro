FUNCTION coord_at_time, initial_xy = initial_xy, initial_time = initial_time, target_time = target_time

target_coord = FINDGEN(SIZE(target_time, /N_ELEMENTS), 2)

FOREACH t, target_time, index DO BEGIN
  daylong = anytim('2017-09-03 00:00:00')-anytim('2017-09-02 00:00:00')
  ll=arcmin2hel(initial_xy(0)/60., initial_xy(1)/60., date = anytim(t, /yohkoh, /date))
  lat = ll(0)
  lon = ll(1)
  lonn = lon + diff_rot((anytim(target_time) - anytim(initial_time)) / daylong, lat)
  new_coord = ROUND(hel2arcmin(lat, lonn, date = anytim(t, /yohkoh, /date))*60.)
  print,'ta',target_coord[index,*],'o',new_coord
  target_coord[index, *] = new_coord
ENDFOREACH

RETURN, target_coord
END