;https://www.lmsal.com/~aschwand/software/ << LOOPS TRACING AND OTHER TUTORIALS

;start time
tstart = '2017-09-03 22:00:00' ; CORRECT SETUP '2017-09-03 22:00:00'
;end_time
tend = '2017-09-04 09:11:00' ; CORRECT SETUP
;step_time
tstep = '00:12:00'
;initial_conditions
centre = [15.,-270.]; at 22:00
out_dir = 'C:\AppsData\gx_out'
tmp_dir = 'C:\AppsData\gx_temp'
dx_km=400.
size_pix=[400,400,75]


;download_serial_hmi, tstart = tstart, tend = tend, tstep = tstep, centre = centre, $
;                      out_dir = out_dir, tmp_dir = tmp_dir, dx_km = dx_km, size_pix = size_pix


DATAFILE ='AIA_94NORH_NLFFFE_170903_225842.sav' ;input image file
LOOPFILE ='LIKE.dat' ;filename for output data
;IMAGE1 =READFITS(datafile,header)
restore,DATAFILE
IMAGE1 = mapw.data
NSM1 =3 ;lowpass filter
RMIN =30 ;minimum curvature radius of loop (pixels)
LMIN =25 ;minimum loop length (in pixels)
NSTRUC =1000 ;maximum limit of traced structures used in array dimension
NLOOPMAX =1000 ;maximum number of detected loops
NGAP =0 ;number of pixels in loop below flux threshold (0,...3)
THRESH1 =0.0 ;ratio of image base flux to median flux
THRESH2 =10 ;threshold in detected structure (number of significance ;levels with respect to the median flux in loop profile
TEST =1001 ;option for display of traced structures if TEST < NSTRUC
PARA =[NSM1,RMIN,LMIN,NSTRUC,NLOOPMAX,NGAP,THRESH1,THRESH2]
LOOPTRACING_AUTO4,IMAGE1,IMAGE2,LOOPFILE,PARA,OUTPUT,TEST
x = IMAGE(IMAGE2)

END