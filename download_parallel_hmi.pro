; THIS PRO NEEDS TO BE IMPLEMENTED LATER. DOESN'T WORK AS EXPECTED ;;;

PRO download_parallel_hmi, tstart = tstart, tend = tend, tstep = tstep, centre = centre, $
  size_pix = size_pix, dx_km = dx_km, out_dir = out_dir, tmp_dir = tmp_dir, n_proc = n_proc
compile_opt idl2, logical_predicate
  ;
  ; THIS IS ROUTINE TO DOWNLOAD HMI AND AIA DATA IN SERIAL MANNER AND CREATE POTENTIAL FIELD BOXES
  ;
  ; 1 arcsec = 727 km
  ; HMI resolution = 1 arcsec, Pixel size = 1/2 arcsec
  ;initial_conditions

  ;computing_count_of_tasks

  cot = ROUND((anytim(tend)-anytim(tstart))/anytim(tstep))
  
  print, 'count_of_tasks =', cot
  
  anytstep = anytim(tstep)

  indices = INDGEN(cot)

  times = anytim(tstart) + indices * anytstep

  centres = coord_at_time(initial_xy = centre, initial_time = tstart, target_time = times)
  
  IF (cot MOD n_proc) GT 0 THEN tasks_per_proc = FLOOR(cot / n_proc) + 1 $
  ELSE tasks_per_proc = ROUND(cot / n_proc)
    
  ; times per process
  t_proc = INDGEN(n_proc, tasks_per_proc)
  
  ; centres per process
  c_proc = INDGEN(n_proc, tasks_per_proc)
    
  ; all exclude one proc
  FOR i = 0, n_proc - 2  DO BEGIN
    t_proc[i, *] = times[i*tasks_per_proc: (i+1)*tasks_per_proc - 1]
    c_proc[i, *] = centres[i*tasks_per_proc: (i+1)*tasks_per_proc - 1]
  ENDFOR
  
  ; last processor
  t_last = times[(n_proc-1) * tasks_per_proc: *]
  c_last = centres[(n_proc-1) * tasks_per_proc: *]
  
  
  tasks = OBJARR(n_proc)
  FOR i = 0, n_proc - 2 DO BEGIN
    print,i
    tasks[i] = OBJ_NEW('IDL_IDLBridge')
    tasks[i]->SetVar,'dir','C:\AppsData'
    tasks[i]->Execute,'cd, dir'
    tasks[i]->SetVar,'!path', !path
    tasks[i]->SetVar,'centres', c_proc[i,*]
    tasks[i]->SetVar,'times', t_proc[i,*]
    tasks[i]->SetVar,'size_pix', size_pix
    tasks[i]->SetVar,'dx_km', dx_km
    ;tasks[i]->SetVar,'out_dir', out_dir
    ;tasks[i]->SetVar,'tmp_dir', tmp_dir
    tasks[i]->SetVar,'x', 0
    ;tasks[i]->Execute,'print, kek',/NOWAIT
    tasks[i]->Execute,'x = make_boxes, times = times, centres = centres, size_pix = size_pix, dx_km = dx_km, out_dir ='+out_dir+', tmp_dir = ' + tmp_dir, /NOWAIT
    print,tasks[i]->Status()
  ENDFOR
  ; LAST PROC
  tasks[n_proc-1] = OBJ_NEW('IDL_IDLBridge')
  tasks[n_proc-1]->SetVar,'dir','C:\AppsData'
  tasks[n_proc-1]->Execute,'cd, dir'
  tasks[n_proc-1]->SetVar,'!path', !path
  tasks[n_proc-1]->SetVar,'centres', c_last
  tasks[n_proc-1]->SetVar,'times', t_last
  tasks[n_proc-1]->SetVar,'size_pix', size_pix
  tasks[n_proc-1]->SetVar,'dx_km', dx_km
  tasks[n_proc-1]->SetVar,'out_dir', out_dir
  tasks[n_proc-1]->SetVar,'tmp_dir', tmp_dir
  tasks[n_proc-1]->SetVar,'x', 0
  tasks[n_proc-1]->Execute,'x = make_boxes, times = times, centres = centres, size_pix = size_pix, dx_km = dx_km, out_dir = out_dir, tmp_dir = tmp_dir'

  
  
END