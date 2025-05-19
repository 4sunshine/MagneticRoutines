pro extract_and_update_amrvacpar, out_dir, example_amrvac, fmt_xmin1, fmt_xmin2, fmt_xmin3, fmt_xmax1, fmt_xmax2, fmt_xmax3, nx, ny

  ; Open input file for reading
  get_lun, lun_in
  openr, lun_in, example_amrvac
  
  ; COPY MOD_USR
  dirname = FILE_DIRNAME(example_amrvac)
  mod_usr_file = dirname + '/mod_usr.t'
  spawn,'cp '+mod_usr_file+ ' /' + out_dir
  
  ; Open output file for writing
  openw, lun_out, out_dir + '/amrvac.par', /get_lun
  
  ; Process each line
  while ~ EOF(lun_in) do begin
    line = ''
    ; Read one line at a time
    readf, lun_in, line
    cur_line = line
    
    case 1 of
      STRMATCH(line, '*xprobmin1=*'): begin
        cur_line = fmt_xmin1
      end
      STRMATCH(line, '*xprobmin2=*'): begin
        cur_line = fmt_xmin2
      end
      STRMATCH(line, '*xprobmin3=*'): begin
        cur_line = fmt_xmin3
      end
      STRMATCH(line, '*xprobmax1=*'): begin
        cur_line = fmt_xmax1
      end
      STRMATCH(line, '*xprobmax2=*'): begin
        cur_line = fmt_xmax2
      end
      STRMATCH(line, '*xprobmax3=*'): begin
        cur_line = fmt_xmax3
      end
      STRMATCH(line, '*domain_nx1=*'): begin
        cur_line = '        domain_nx1='+strtrim(string(nx),2)
      end
      STRMATCH(line, '*domain_nx2=*'): begin
        cur_line = '        domain_nx2='+strtrim(string(ny),2)
      end
      STRMATCH(line, '*domain_nx3=*'): begin
        cur_line = '        domain_nx3='+strtrim(string(ny),2)
      end
      else: cur_line = line ; dummy
    endcase
    
    printf, lun_out, cur_line
  endwhile
  
  ; Close file
  free_lun, lun_in
  free_lun, lun_out
  
end
