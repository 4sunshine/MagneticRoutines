undefine, box
box_file = DIALOG_PICKFILE(TITLE = 'NLFFE BOX SELECT')
norh_file = DIALOG_PICKFILE(TITLE = 'Select NoRH fits file')
restore, box_file
gx_box_add_refmap_norh, box, norh_file, id = 'ipa'
save, box, FILENAME = 'NoRH_' + FILE_BASENAME(box_file)
undefine, box
END
