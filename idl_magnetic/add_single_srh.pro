undefine, box
box_file = DIALOG_PICKFILE(TITLE = 'NLFFE BOX SELECT')
srh_file = DIALOG_PICKFILE(TITLE = 'Select SRH fits file')
restore, box_file

gx_box_add_refmap_srh, box, srh_file, id = 'srh'
save, box, FILENAME = 'SRH_' + FILE_BASENAME(box_file)
undefine, box
END
