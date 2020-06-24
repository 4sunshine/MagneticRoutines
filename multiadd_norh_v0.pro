undefine, box
file = DIALOG_PICKFILE(TITLE = 'NLFFE BOX SELECT', /MULTIPLE_FILES)
sz = size(file)
FOR i = 0, sz[1]-1 DO BEGIN
  restore, file[i]
  norh_file_root = 'C:\AppsData\ipas\'
  fileA = norh_file_root + 'ipa' + STRMID(FILE_BASENAME(file[i]), 7, 13)
  fileS = norh_file_root + 'ips' + STRMID(FILE_BASENAME(file[i]), 7, 13)
  gx_box_add_refmap_norh, box, fileA, id = '17 GHz (R+L), K'
  gx_box_add_refmap_norh, box, fileS, id = '17 GHz (R-L), K'

  save, box, FILENAME = 'NORH_' + FILE_BASENAME(file[i])
  ; FILE_MOVE, FILE_BASENAME(file[i]), 'XDONE_' + FILE_BASENAME(file[i])
  undefine, box
ENDFOR

END