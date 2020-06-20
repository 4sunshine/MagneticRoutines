PRO multi_nlfffe

UNDEFINE,box
files = DIALOG_PICKFILE(TITLE = 'NLFFFE BOXES SELECT', /MULTIPLE_FILES)
sz = size(files)

FOR i = 0, sz[1]-1 DO BEGIN
  restore, files[i]
  print, 'task', i+1, ' of', sz[1]
  return_code = gx_box_make_nlfff_wwas_field("C:\ssw\packages\gx_simulator\gxbox\WWNLFFFReconstruction.dll", box)
  
  cacheSTR = anytim(box.index.DATE_OBS,/hxrbs,/date)
  nameSTR = STRMID(cacheSTR, 0, 2) + STRMID(cacheSTR, 3, 2) + STRMID(cacheSTR, 6, 2) + '_'

  cacheSTR = anytim(box.index.DATE_OBS,/hxrbs,/time)
  nameSTR = nameSTR + STRMID(cacheSTR, 0, 2) + STRMID(cacheSTR, 3, 2) + STRMID(cacheSTR, 6, 2)

  save, box, FILENAME = 'NLFFFE_' + nameSTR + '.sav'
    
ENDFOR



END