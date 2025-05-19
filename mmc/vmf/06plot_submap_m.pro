
IN_DIR='/home/sunshine/data/20170906_090000_TAI'
restore,IN_DIR+'/mapbxyz_m.sav',/ver
xrange = [-244., 44]
yrange = [-395., -155.5]
res = [0.5,0.5]
sub_map,mapbx,smapbx,xrange=xrange,yrange=yrange
dsmapbx = drot_map(smapbx,time = smapbx.time,resolution=res,/preserve_area)
smapbx = dsmapbx
sub_map,mapby,smapby,xrange=xrange,yrange=yrange
dsmapby = drot_map(smapby,time = smapby.time,resolution=res,/preserve_area)
smapby = dsmapby
sub_map,mapbz,smapbz,xrange=xrange,yrange=yrange
dsmapbz = drot_map(smapbz,time = smapbz.time,resolution=res,/preserve_area)
smapbz = dsmapbz
wdef,1,1300,800
!p.background=255
smapbz.id='SDO/HMI Heliographic Bz'

plot_map,smapbz,color=0,bcolor=0,charsize=2.0,dmax=2000,dmin=-2000
write_png,IN_DIR+'/sbz_m.png',tvrd()
smapbz.id='Heliographic Bxyz'
print,'Flux balance coefficient:',total(smapbz.data)/total(abs(smapbz.data))
plot_map,smapbz,color=0,bcolor=0,charsize=2.0,dmax=2000,dmin=-2000
plot_vmap,/over,smapbx,smapby,mapbz=smapbz,limit=150,scale=0.01,iskip=15,jskip=15,$
          v_color=255,axis_color=0,/Nolabels,v_thick=2.0,/Noaxis
!p.background=0
write_png,IN_DIR+'/sbxyz_m.png',tvrd()

; CHECK HERE THE SHAPE OF DATA. IT SHOULD BE MULTIPLE OF A GOOD NUMBER * 4
; Select the RANGE in the begin of file.
help, smapbz.data
; stop
save,smapbx,smapby,smapbz,filename=IN_DIR+'/smapbxyz_m.sav'

end
