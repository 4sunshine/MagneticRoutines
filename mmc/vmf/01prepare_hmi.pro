; pro prepare_hmi 
; 

DATA_DIR = '/home/sunshine/Downloads'
EVENT_TIME = '20170906_090000_TAI' ; Actual time
REFERENCE_TIME = '20170903_090000_TAI'  ; The time to DROT for
SAVE_ALL_DIR = '/home/sunshine/data'
IF ~FILE_TEST(SAVE_ALL_DIR, /DIRECTORY) THEN FILE_MKDIR, OUT_DIR
OUT_DIR = SAVE_ALL_DIR + '/' + EVENT_TIME
IF ~FILE_TEST(OUT_DIR, /DIRECTORY) THEN FILE_MKDIR, OUT_DIR

fn1 = DATA_DIR+'/hmi.B_720s.' + EVENT_TIME + '.field.fits'
read_sdo,fn1,index1,data1
index2map,index1,data1,mapb
fn2 =DATA_DIR+'/hmi.B_720s.'+ EVENT_TIME +'.inclination.fits'
read_sdo,fn2,index2,data2
index2map,index2,data2,mapi
fn3=DATA_DIR+'/hmi.B_720s.'+EVENT_TIME+'.azimuth.fits'
read_sdo,fn3,index3,data3
fn4=DATA_DIR+'/hmi.B_720s.'+EVENT_TIME+'.disambig.fits'
read_sdo,fn4,index4,data4
ns=size(data4)
for j=0,ns[2]-1 do begin
  for i=0,ns[1]-1 do begin
    if (data4[i,j] gt 3) then begin
      data3[i,j]=data3[i,j]+180.0
    endif
  endfor
endfor
index2map,index3,data3,mapa

mapbz = mapb
mapbz.data = rotate(mapb.data*cos(mapi.data*!dtor),2)
mapbz.roll_angle = 0.0

; FIND CENTER HERE
dx0 = 0.5*(956.5 - 949.5)
dy0 = 0.5*(950.0 - 953.0)
mapbz.xc = mapbz.xc - dx0         ;Alignment by comparing the position of the solar limbs
mapbz.yc = mapbz.yc - dy0
mapbx = mapb
mapbx.data = rotate(mapb.data*sin(mapi.data*!dtor)*cos((mapa.data + 270.0)*!dtor),2)  ;The azimuth angle is measured from the CCD+y direction, which is the south, since the solar P angle is ~180 degree
mapbx.roll_angle = 0.0
index = where(mapbx.data le -10000.0)
mapbx.data[index] = 0.0
mapbx.xc = mapbz.xc
mapbx.yc = mapbz.yc
mapby = mapb
mapby.data = rotate(mapb.data*sin(mapi.data*!dtor)*sin((mapa.data + 270.0)*!dtor),2)  ;The azimuth angle is measured from the CCD+y direction, which is the south, since the solar P angle is ~180 degree
mapby.roll_angle = 0.0
mapby.data[index] = 0.0
mapby.xc = mapbz.xc
mapby.yc = mapbz.yc

intimei = utc2int(mapbz.time)
mjdi  = intimei.mjd + fix((intimei.time + 120000.0)/(24.0*3600.0*1000.0))
timei = (intimei.time + 120000.0) mod (24.0*3600.0*1000.0)
intimei.mjd = mjdi
intimei.time= timei
utctime = anytim2utc(intimei,/vms)
mapbz.time = utctime
mapbx.time = utctime
mapby.time = utctime

IF (REFERENCE_TIME NE EVENT_TIME) THEN BEGIN
  ; Load smapbz from sav file
  restore, SAVE_ALL_DIR + '/' + REFERENCE_TIME + '/bxyz_submap.sav',/ver
  mapbz = drot_map(mapbz, time=smapbz.time)
  mapbz.time = utctime
  mapbz.rtime = utctime
  mapby = drot_map(mapby, time=smapby.time)
  mapby.time = utctime
  mapby.rtime = utctime
  mapbx = drot_map(mapbx, time=smapbx.time)
  mapbx.time = utctime
  mapbx.rtime = utctime
  smapbx = {null: 0}
  smapby = {null: 0}
  smapbz = {null: 0}
ENDIF

loadct,0
wdef,1,800,800
!p.background = 255
plot_map,mapbz,dmax=2000,dmin=-2000,color=0,charsize=1.8
write_png,OUT_DIR+'/bz.png',tvrd()
wdef,2,800,800
plot_map,mapbz,dmax=2000,dmin=-2000,color=0,charsize=1.8
plot_vmap,/over,mapbx,mapby,mapbz=mapbz,limit=50,scale=0.1,iskip=30,jskip=30,$
  v_color=255,axis_color=0,/Nolabels,v_thick=2.0,/No_arrow_head  ;,/Noaxis
write_png,OUT_DIR+'/bxyz.png',tvrd()
; DRAW A SUBREGION HERE. USED FOR SOLAR LIMB COORDINATES FINDING
sub_map,mapbz,smapbz,xrange=[-250,50.0],yrange=[-400,-100.0]
print,'Flux balance coefficient:',total(smapbz.data)/total(abs(smapbz.data))
sub_map,mapbx,smapbx,ref_map=smapbz
sub_map,mapby,smapby,ref_map=smapbz
wdef,1,800,800
plot_map,smapbz,dmax=2000,dmin=-2000,color=0,charsize=1.8
write_png,OUT_DIR+'/sbz.png',tvrd()
wdef,2,800,800
plot_map,smapbz,dmax=2000,dmin=-2000,color=0,charsize=1.8
plot_vmap,/over,smapbx,smapby,mapbz=smapbz,limit=180,scale=0.012,iskip=15,jskip=15,$
  v_color=255,axis_color=0,/Nolabels,v_thick=2.0 ;,/No_arrow_head  ;,/Noaxis
write_png,OUT_DIR+'/sbxyz.png',tvrd()
i=0

save,smapbx,smapby,smapbz,filename=OUT_DIR+'/bxyz_submap.sav'

end