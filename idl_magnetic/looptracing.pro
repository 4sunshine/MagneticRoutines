PRO LOOPTRACING

;https://www.lmsal.com/~aschwand/software/ << LOOPS TRACING AND OTHER TUTORIALS

files_extension = '.sav'
files = DIALOG_PICKFILE(TITLE = 'SELECT FILES TO TRACE', FILTER = '*' + files_extension, /MULTIPLE_FILES)
sz = size(files)
target_dir = 'loops'
  FOR i = 0, sz[1]-1 DO BEGIN
    IF i EQ 0 THEN BEGIN
      target_dir = FILE_DIRNAME(files[0]) + '\' + target_dir
      print, target_dir
      FILE_MKDIR, target_dir
    ENDIF
    ;IMAGE1 =READFITS(datafile,header)
    print, 'task', i + 1 ,'of ', sz[1]
    restore, files[i]
    LOOPFILE = FILE_BASENAME(files[i], files_extension)
    LOOPFILE = target_dir + '\traces' + LOOPFILE + '.dat'
    print, LOOPFILE
    IMAGE1 = mapw.data
    NSM1 =3 ;lowpass filter
    RMIN =30 ;minimum curvature radius of loop (pixels)
    LMIN =25 ;minimum loop length (in pixels)
    NSTRUC =1000 ;maximum limit of traced structures used in array dimension
    NLOOPMAX =10 ;maximum number of detected loops
    NGAP =3 ;number of pixels in loop below flux threshold (0,...3)
    THRESH1 =0.0 ;ratio of image base flux to median flux
    THRESH2 =10 ;threshold in detected structure (number of significance ;levels with respect to the median flux in loop profile
    TEST =1001 ;option for display of traced structures if TEST < NSTRUC
    PARA =[NSM1,RMIN,LMIN,NSTRUC,NLOOPMAX,NGAP,THRESH1,THRESH2]
    LOOPTRACING_AUTO4,IMAGE1,IMAGE2,LOOPFILE,PARA,OUTPUT,TEST
   ENDFOR
END