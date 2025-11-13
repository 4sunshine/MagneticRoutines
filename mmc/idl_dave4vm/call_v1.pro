
pro call_v1


;restoring the save
restore, '/usr/users/UNN/w17016451/saves/dave4vmparameters.save',/v


;do_dave4vm_and, bx_start, by_start, bz_start, bx_stop, by_stop, bz_stop, $
;                     DX, DY, DT, vel4vm, magvm, windowsize = windowsize


;defining dx and dy
;value for dx is a test.
dx=1e3
dy=dx


bx_start = bp_start(*,*,0)
by_start = bt_start(*,*,0)
bz_start = br_start(*,*,0)
bx_stop = bp_stop(*,*,0)
by_stop = bt_stop(*,*,0)
bz_stop = br_stop(*,*,0)
DTi = DT(0) 

;calling
do_dave4vm_and, bx_start, by_start, bz_start, bx_stop, by_stop, bz_stop,$
		DX, DY, DTi, vel4vm, magvm, windowsize = 10 ; windowsize = windowsize (default)



stop

;the lines below are not really necessary...this can be done via:
;tvscl, /nan, variable.(tag)


;testing getting rid of nans
a = vel4vm.(3)
b = WHERE(FINITE(A, /NAN))

n0 = 0
for i = 0, 446978 DO BEGIN

;trying for mean instead of 0
a(b) = -0.063856294

ENDFOR

stop

END
