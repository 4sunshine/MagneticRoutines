;https://www.lmsal.com/~aschwand/software/ << LOOPS TRACING AND OTHER TUTORIALS

; Data-driven Modeling of a Coronal Magnetic Flux Rope: from Birth to Death, 2023
; dx_km = 367.084 : 560 pixels : -102600, 102600 km range
; dy_km = 367.084 : 400 pixels : -73233, 73233 km range
; dz_km = 367.084 : 560 pixels : 1000, 205200 km range

;start time
tstart = '2017-09-03 09:00:00' ; CORRECT SETUP '2017-09-03 22:00:00'
;end_time
tend = '2017-09-03 09:14:00' ; CORRECT SETUP
;step_time
tstep = '00:12:00'
;initial_conditions
centre = [218.,174.]; at 22:00

out_dir = 'C:\AppsData\gx_out'
tmp_dir = 'C:\AppsData\gx_temp'

dx_km=367.084
size_pix=[560,400,560]

download_serial_hmi_by_list, size_pix = size_pix, dx_km = dx_km, out_dir = out_dir, tmp_dir = tmp_dir

;download_serial_hmi, tstart = tstart, tend = tend, tstep = tstep, centre = centre, $
;                      out_dir = out_dir, tmp_dir = tmp_dir, dx_km = dx_km, size_pix = size_pix


;DATAFILE ='C:\AppsData\NLFFFE\BASEMAPS\sav\17 GHz (R+L), KNORH_NLFFFE_170903_224642.sav' ;input image file
;LOOPFILE ='NORH_I.dat' ;filename for output data
;IMAGE1 =READFITS(datafile,header)
;restore,DATAFILE
;IMAGE1 = mapw.data
;NSM1 =3 ;lowpass filter
;RMIN =30 ;minimum curvature radius of loop (pixels)
;LMIN =25 ;minimum loop length (in pixels)
;NSTRUC =1000 ;maximum limit of traced structures used in array dimension
;NLOOPMAX =1000 ;maximum number of detected loops
;NGAP =0 ;number of pixels in loop below flux threshold (0,...3)
;THRESH1 =0.0 ;ratio of image base flux to median flux
;THRESH2 =10 ;threshold in detected structure (number of significance ;levels with respect to the median flux in loop profile
;TEST =1001 ;option for display of traced structures if TEST < NSTRUC
;PARA =[NSM1,RMIN,LMIN,NSTRUC,NLOOPMAX,NGAP,THRESH1,THRESH2]
;LOOPTRACING_AUTO4,IMAGE1,IMAGE2,LOOPFILE,PARA,OUTPUT,TEST
;x = IMAGE(IMAGE2)


END