function dave4vm, mag, window_size, $
  sv = sv, sigma2 = sigma2, chi2 = chi2, errors = errors, $
  threshold = threshold, noise = noise, double = double, am = AM, $
  float = float, missing_value = missing_value, help = help, $
  kernel = kernel
  compile_opt idl2
  ;
  ; on_error,2 ; Return to the caller of the program unit that established the ON_ERROR condition.
  ;
  nparms = n_params()
  ;
  if keyword_set(help) then begin
    info:
    message, $
      '============================================================================' $
      , /info
    message, '', /info
    message, 'Usage:', /info
    message, 'IDL>vel=dave4vm(mag,window_size,$', /info
    message, 'IDL>sv=sv,sigma2=sigma2,chi2=chi2,errors=errors,threshold=threshold,$', /info
    message, 'IDL>noise=noise,double=double,missing_value=missing_value)', /info
    message, '', /info
    message, $
      'Title: DAVE4VM - Differential Affine Velocity Estimator', /info
    message, $
      '                                                   for Vector Magnetograms', /info
    message, '', /info
    message, 'Computes plasma velocities from a structure of magnetic field measurements', /info
    message, '', /info
    message, 'Please reference:', /info
    message, 'Schuck, P. W., Tracking Vector Magnetograms with the Magnetic Induction', /info
    message, '              Equation, Submitted to ApJ, 2008.', /info
    message, '', /info
    message, 'Author: Peter W. Schuck', /info
    message, 'schuck@ppdmail.nrl.navy.mil', /info
    message, 'Plasma Physics Division', /info
    message, 'United States Naval Research Laboratory', /info
    message, 'Washington, DC, 20375', /info
    message, '', /info
    message, 'VERSION HISTORY', /info
    message, 'VERSION 1.0 written: 07-11-2006 "dave_vm"', /info
    message, 'VERSION 2.0 written: 01-15-2008 "dave_vm"', /info
    message, 'VERSION 2.1 written: 01-31-2008 "dave4vm"', /info
    message, '', /info
    message, 'INPUT:', /info
    message, '        MAG - structure of vector magnetic field measurements', /info
    message, '              MAG.DX    X spatial scale (used to compute B?X)', /info
    message, '              MAG.DY    Y spatial scale (used to compute B?Y)', /info
    message, '                        ? = X Y Z', /info
    message, '                                                           ', /info
    message, '              MAG.BZT   Array[NX,NY]  time derivative of Bz', /info
    message, '              MAG.BX    Array[NX,NY]  X component of B (Bx)', /info
    message, '              MAG.BXX   Array[NX,NY]  X derivative of Bx', /info
    message, '              MAG.BXY   Array[NX,NY]  Y derivative of By', /info
    message, '              MAG.BY    Array[NX,NY]  Y component of B (By)', /info
    message, '              MAG.BYX   Array[NX,NY]  X derivative of By', /info
    message, '              MAG.BYY   Array[NX,NY]  Y derivative of By', /info
    message, '              MAG.BZ    Array[NX,NY]  Z component of B (Bz)', /info
    message, '              MAG.BZX   Array[NX,NY]  X derivative of Bz', /info
    message, '              MAG.BZY   Array[NX,NY]  Y derivative of Bz', /info
    message, '', /info
    message, 'WINDOW_SIZE - A one or two element vector for the window aperture', /info
    message, '', /info
    message, 'OUTPUT:', /info
    message, '        VEL - Array[NX,NY] of structures of coefficients', /info
    message, '              U0        X-Velocity ', /info
    message, '              V0        Y-Velocity ', /info
    message, '              W0        Z-Velocity ', /info
    message, '              UX        Local X derivative of the X-Velocity', /info
    message, '              VX        Local X derivative of the Y-Velocity', /info
    message, '              WX        Local X derivative of the Z-Velocity', /info
    message, '              UY        Local Y derivative of the X-Velocity', /info
    message, '              VY        Local Y derivative of the Y-Velocity', /info
    message, '              WY        Local Y derivative of the Z-Velocity', /info
    message, '              WINDOW_SIZE Local window size', /info
    message, '', /info
    dave_keywords
    message, '', /info
    message, '***************************************************************', /info
    message, '', /info
    message, 'Important! Velocities must be orthogonalized to obtain plasma velocities ', /info
    message, 'perpendicular to the magnetic field!', /info
    message, '', /info
    message, '***************************************************************', /info
    message, '', /info
    message, 'AUTHORIZATION TO USE AND DISTRIBUTE', /info
    message, 'I hereby agree to the following terms governing the use and', /info
    message, 'redistribution of the DAVE4VM software release originally', /info
    message, 'written and developed by Dr. P. W. Schuck', /info

    message, '', /info
    message, 'Redistribution and use in source and binary forms, with or', /info
    message, 'without modification, are permitted provided that (1) source', /info
    message, 'code distributions retain this paragraph in its entirety, (2)', /info
    message, 'distributions including binary code include this paragraph in', /info
    message, 'its entirety in the documentation or other materials provided', /info
    message, 'with the distribution, (3) improvements, additions and', /info
    message, 'upgrades to the software will be provided to NRL Authors in', /info
    message, 'computer readable form, with an unlimited, royalty-free', /info
    message, 'license to use these improvements, additions and upgrades', /info
    message, 'and the authority to grant unlimited royalty-free sublicenses', /info
    message, 'to these improvements and (4) all published research using', /info
    message, 'this software display the following acknowledgment', /info
    message, '``This work uses the DAVE/DAVE4VM codes written and developed', /info
    message, 'by the Naval Research Laboratory.''', /info
    message, '', /info
    message, 'Neither the name of NRL or its contributors, nor any entity', /info
    message, 'of the United States Government may be used to endorse or', /info
    message, 'promote products derived from this software, nor does the', /info
    message, 'inclusion of the NRL written and developed software directly', /info
    message, 'or indirectly suggest NRL''s or the United States Government''s', /info
    message, 'endorsement of this product.', /info
    message, '', /info
    message, 'THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS', /info
    message, 'OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE', /info
    message, 'IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A', /info
    message, 'PARTICULAR PURPOSE.', /info

    message, '', /info
    message, '***************************************************************', /info
    message, $
      '============================================================================', /info
    return, -1
  endif
  ;
  if (nparms ne 2) then begin
    message, 'Incorrect number of parameters', /info
    message, 'for help use:', /info
    message, 'IDL> dave4vm,/help'
  endif
  ;
  ;
  ; check keywords
  if ~keyword_set(errors) then errors = 0
  ;
  if ~keyword_set(noise) then noise = 0.d0
  ;
  if ~keyword_set(double) then double = 0 else double = 1
  if ~keyword_set(float) then float = 0 else float = 1

  double = 1
  float = 0
  ;
  if (float eq double) then begin
    if (float eq 0) then begin
      float = (size(mag.bz, /type) eq 4)
      double = (size(mag.bz, /type) eq 5)
    endif else begin
      message, 'both keywords "DOUBLE" and "FLOAT" cannot be set', /info
      message, 'for help use:', /info
      message, 'IDL> dave4vm,/help'
    endelse
  endif
  ;
  if ~keyword_set(missing_value) then $
    ; if (double) then missing_value=!values.D_NAN $
    ; else missing_value=!values.F_NAN
    if (double) then missing_value = !values.d_nan $
    else missing_value = !values.f_nan
  ;
  if ~keyword_set(threshold) then threshold = 1.d0

  print, keyword_set(missing_value), missing_value
  print, threshold
  double = 1
  float = 0
  ;
  NWS = n_elements(window_size)
  if ((NWS lt 1) or (NWS gt 2)) then begin
    message, 'WINDOW_SIZE must be one or two dimensional', /info
    message, 'for help use:', /info
    message, 'IDL> dave4vm,/help'
  endif
  ;
  MAG_LIST = ['BX', 'BY', 'BZ', 'BXX', 'BYX', 'BZX', 'BXY', 'BYY', 'BZY', 'BZT']

  all_exist = 1 ; Start assuming all exist

  foreach tag, MAG_LIST do begin
    if ~tag_exist(mag, tag) then begin
      all_exist = 0
      break
    endif
  endforeach

  ; if (check_tags(mag_list,mag,/knot,/verbose)) then goto,info
  ;
  ; define arrays
  sz = size(mag.bz)
  if (sz[0] ne 2) then begin
    message, 'Image arrays must be two dimensional', /info
    message, 'for help use:', /info
    message, 'IDL> dave4vm,/help'
  endif
  ;
  ; define arrays
  vel = missing_value
  dum = replicate(vel, sz[1], sz[2])
  v = {u0: dum, ux: dum, uy: dum, $
    v0: dum, vx: dum, vy: dum, $
    w0: dum, wx: dum, wy: dum}

  sv = make_array(9, sz[1], sz[2], double = double, float = float)
  WW = make_array(9, 9, double = double, float = float)
  chi2 = make_array(sz[1], sz[2], double = double, float = float)
  sigma2 = make_array(9, 9, sz[1], sz[2], double = double, float = float)
  ;
  chi2[*] = missing_value
  sigma2[*] = missing_value
  aperture = replicate(255b, sz[1], sz[2])
  ;
  id = indgen(9)
  ;
  ; *** NOTE IMPORTANT! ********
  ; it is necessary for the integer (pixel) values of X and Y to
  ; be multiplied by dx and dy respectively to ensure rescaling
  ; of derivatives leads to the identical answer
  ;
  ; construct weighting functions
  nw = fix(2 * fix(window_size / 2) + 1)
  if (n_elements(nw) eq 1) then nw = [nw[0], nw[0]]
  x = rebin((lindgen(nw[0]) - nw[0] / 2), nw[0], nw[1]) * mag.dx
  y = transpose(rebin((lindgen(nw[1]) - nw[1] / 2), nw[1], nw[0])) * mag.dy
  ;
  psf = make_array(nw[0], nw[1], double = double, float = float)
  ; default window is top-hat
  psf[*] = 1.d0
  ; normalize
  psf[*] = psf / total(psf, /double)
  ; moments
  psfx = psf * x
  psfy = psf * y
  psfxx = psf * x ^ 2
  psfyy = psf * y ^ 2
  psfxy = psf * x * y
  ;
  v = add_tag(v, nw, 'WINDOW_SIZE')
  ;
  AM = dave4vm_matrix(mag.bx, mag.bxx, mag.bxy, $
    mag.by, mag.byx, mag.byy, $
    mag.bz, mag.bzx, mag.bzy, $
    mag.bzt, psf, psfx, psfy, psfxx, psfyy, psfxy, $
    double = double, float = float)
  ;
  kernel = {psf: psf, psfx: psfx, psfy: psfy, psfxx: psfxx, psfyy: psfyy, psfxy: psfxy}
  ;
  ; estimate trace
  trc = total((reform(AM, 100, sz[1], sz[2]))[id * 10 + id, *, *], 1, /double)
  ;
  ; find locations where the aperture problem could be resolved
  index = where(trc gt threshold, N)
  ;
  if (N eq 0) then begin
    message, 'The input images MAG.Bx are pathological or the window', /info
    message, 'size SIGMA is too small. The aperture problem cannot be', /info
    ; 'resolved.',level=-1
    return, -1
  endif
  ;

  ; loop over good pixels
  for ii = 0l, N - 1l do begin
    j = index[ii] / sz[1]
    i = index[ii] mod sz[1]
    ;
    AA = AM[*, *, i, j]
    GA = AA[0 : 8, 0 : 8]
    FA = -reform(AA[0 : 8, 9], 9)
    DESIGN = GA
    SOURCE = FA
    ;
    ; SVDC,DESIGN,W,UA,VA
    ; SV[*,i,j]=W[sort(W)]
    ; mn=0.d0
    ; zero=where(W le mn)
    ; if (zero[0] ne -1) then W[zero]=0.d0
    ; VECTOR=SVSOL(UA,W,VA,SOURCE,/double)
    ;
    ; USE divide-and-conquer algorithm
    ;
    ; rcondition=1.d-6
    vector = la_least_squares(DESIGN, SOURCE, double = double, $
      method = 3, residual = chisq, rcondition = rcondition)
    ;
    v.u0[i, j] = vector[0]
    v.v0[i, j] = vector[1]
    v.ux[i, j] = vector[2]
    v.vy[i, j] = vector[3]
    v.uy[i, j] = vector[4]
    v.vx[i, j] = vector[5]
    v.w0[i, j] = vector[6]
    v.wx[i, j] = vector[7]
    v.wy[i, j] = vector[8]
    ;
  endfor
  ;
  return, v
end

; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
