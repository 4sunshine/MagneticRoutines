PRO download_parallel_hmi, tstart = tstart, tend = tend, tstep = tstep, centre = centre, $
  size_pix = size_pix, dx_km = dx_km, out_dir = out_dir, tmp_dir = tmp_dir, nproc = nproc
  ;
  ; THIS IS ROUTINE TO DOWNLOAD HMI AND AIA DATA IN SERIAL MANNER AND CREATE POTENTIAL FIELD BOXES
  ;
  ; 1 arcsec = 727 km
  ; HMI resolution = 1 arcsec, Pixel size = 1/2 arcsec
  ;initial_conditions

  ;computing_count_of_tasks

  cot = ROUND((anytim(tend)-anytim(tstart))/anytim(tstep))
  
  print, 'count_of_tasks =', cot
  
  tasks_per_proc = FLOOR(cot / n_proc) + 1
  
  anytstep = anytim(tstep)
  
  indices = INDGEN(cot)
  
  times = anytim(tstart) + indices * anytstep
  
  centres = coord_at_time(initial_xy = centre, initial_time = tstart, target_time = times)
  
  ; times per process
  t_proc = INDGEN(n_proc)
  
  ; centres per process
  c_proc = INDGEN(n_proc)
  
  ; all exclude one proc
  FOR i = 0, n_proc - 1  DO BEGIN
    t_proc[i] = times[i*tasks_per_proc: (i+1)*tasks_per_proc]
    c_proc[i] = centres[i*tasks_per_proc: (i+1)*tasks_per_proc]
  ENDFOR
  
  ; last processor
  t_proc[-1] = times[(n_proc-1) * tasks_per_proc: ]
  c_proc[-1] = centres[(n_proc-1) * tasks_per_proc: ]
  
  
  tasks = OBJARR(n_proc)
  FOR i = 0, n_proc - 1 DO BEGIN
    tasks[i] = IDL_IDLbridge()
    tasks[i]->SetVar,'dir','C:\AppsData'
    tasks[i]->Execute,'cd, dir'
    tasks[i]->SetVar,'!path', !path
    tasks[i]->SetVar,'centres', c_proc[i]
    tasks[i]->SetVar,'times', t_proc[i]
    tasks[i]->SetVar,'size_pix', size_pix
    tasks[i]->SetVar,'dx_km', dx_km
    tasks[i]->SetVar,'out_dir', out_dir
    tasks[i]->SetVar,'tmp_dir', tmp_dir
    tasks[i]->Execute,'FOREACH t, times, index DO BEGIN gx_box_prepare_box, t, centres[index], size_pix, dx_km, out_dir = out_dir, /cea, tmp_dir = tmp_dir, /aia_euv', /nowait
  ENDFOR
  
END