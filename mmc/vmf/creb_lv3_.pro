; IN_DIR='/home/sunshine/data/20170906_090000_TAI'

pro creb_lv3_, IN_DIR
;creat boundary data for nonlinear force-free field extrapolation. Using "multigrid.pro" with 
;level=3 to make 4*4 bin data.

prep=1
level=3
;if prep equals 1, the boundary data are preprocessed. The MDI data cannot
;be preprocessed, since there are no transverse fields.

file=find_file(IN_DIR+'/smapbxyz*m.sav')
;
nn=n_elements(file)
;nn=1    ;for test
for i=0,nn-1 do begin
  restore,file[i]
  dim=size(smapbz.data,/dim)
  bz=smapbz.data
  print,'Flux Balance Coefficient:',total(bz)/total(abs(bz))
  bx=smapbx.data
  by=smapby.data
  ss=size(bz)
  wdef,1,ss[1],ss[2]
  tvscl,bz
  save,bx,by,bz,filename='temp.sav'
  multigrid,file='temp.sav',level=level,prep=prep
  filename=strmid(smapbz.time,12,2)+strmid(smapbz.time,15,2)+'_lv3' ; OLD naming
  filename='level3_data'
  filename=IN_DIR + '/' + filename
  spawn,'mkdir '+ filename
  spawn,'mv nd.ini '+filename
  spawn,'mv grid.ini '+filename
  spawn,'mv allboundaries.dat '+filename
endfor
spawn,'rm temp.sav'

end
