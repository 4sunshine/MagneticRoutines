; +
;    prepare a structure data for dave4vm
;
;   inputs: bxyz_start: vector field at T_start
;           bxyz_stop: vector field at T_stop
;   output: mag: a data structure to be used in dave4vm.
;
;               -----YL, 2011.02.17
; -

; My changes: added DT as an entry. Removed: t_start, t_stop, vel4vm.
;
;
;
;
;
;

PRO do_dave4vm_and, bx_start, by_start, bz_start, bx_stop, by_stop, bz_stop, $
                     DX, DY, DT, vel4vm, magvm, windowsize = windowsize

if not keyword_set(windowsize) then windowsize = 20

; estimate the optimal windowsize
; should not be less than 11
; compute the numerical derivatives of the flux transport vectors UxBz and UyBz and compare with dBz/dt

;DT = float(t_stop - t_start)
BZT = double((bz_stop - bz_start)/DT)

;     use time_centered spatial variables
BX = (bx_stop + bx_start)/2.
BY = (by_stop + by_start)/2.
BZ = (bz_stop + bz_start)/2.

;     compute 5-point optimized derivatives
odiffxy5,BX,BXX,BXY
odiffxy5,BY,BYX,BYY
odiffxy5,BZ,BZX,BZY

magvm={BZT:BZT,BX:BX,BXX:BXX/DX,BXY:BXY/DY,BY:BY,BYX:BYX/DX,BYY:BYY/DY,$
       BZ:BZ,BZX:BZX/DX,BZY:BZY/DY,DX:DX,DY:DY,DT:DT}

wsize = windowsize
vel4vm = dave4vm(magvm, wsize)

end
