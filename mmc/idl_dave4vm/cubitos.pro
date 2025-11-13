;por alguma razão o nome da rotina muda aqui...efeito puramente cosmético mas atenção à chamada
pro cubitos

fs1=file_search('/usr/users/UNN/w17016451/jsoc.stanford.edu/SUM87/D1009191004/S00000/*.Bp.fits', count=n1)
fs2=file_search('/usr/users/UNN/w17016451/jsoc.stanford.edu/SUM87/D1009191004/S00000/*.Br.fits', count=n2)
fs3=file_search('/usr/users/UNN/w17016451/jsoc.stanford.edu/SUM87/D1009191004/S00000/*.Bt.fits', count=n3)

;print, n1,n2,n3

;lista de variaveis
; RESTORE: Restored variable: BX_START.
; RESTORE: Restored variable: BX_STOP.
; RESTORE: Restored variable: BY_START.
; RESTORE: Restored variable: BY_STOP.
; RESTORE: Restored variable: BZ_START.
; RESTORE: Restored variable: BZ_STOP.
; RESTORE: Restored variable: DX.
; RESTORE: Restored variable: THR.
; RESTORE: Restored variable: VX_START.
; RESTORE: Restored variable: VX_STOP.
; RESTORE: Restored variable: VY_START.
; RESTORE: Restored variable: VY_STOP.
; RESTORE: Restored variable: VZ_START.
; RESTORE: Restored variable: VZ_STOP.
; RESTORE: Restored variable: T_START.
; RESTORE: Restored variable: T_STOP.
;a saida disso vai pro shootouts.pro, seriam necessarios os parametros de velocidade?
;aparentemente eles sao as truths.

Bp_start = dblarr(949,471,20)
Br_start = dblarr(949,471,20)
Bt_start = dblarr(949,471,20)

Bp_stop = dblarr(949,471,20)
Br_stop = dblarr(949,471,20)
Bt_stop = dblarr(949,471,20)

t_start = strarr(20)
t_stop = strarr(20)
dt = strarr(20)
n0=0
FOR i=n0, n1-2 DO BEGIN

read_sdo, fs1(i), indexBp, dataBp, /uncomp_delete
read_sdo, fs2(i), indexBr, dataBr,  /uncomp_delete
read_sdo, fs3(i), indexBt, dataBt,  /uncomp_delete

read_sdo, fs1(i+1), indexBp2, dataBp2, /uncomp_delete
read_sdo, fs2(i+1), indexBr2, dataBr2,  /uncomp_delete
read_sdo, fs3(i+1), indexBt2, dataBt2,  /uncomp_delete

;nesse caso todos tem o mesmo tamanho. 
;pro proposito de testes vamos seguir em frente com os cubos
;help, dataBp
;help, dataBr
;help, dataBt

Bp_start(*,*,i) = dataBp
Br_start(*,*,i) = dataBr
Bt_start(*,*,i) = dataBt

Bp_stop(*,*,i) = dataBp2
Br_stop(*,*,i) = dataBr2
Bt_stop(*,*,i) = dataBt2

;se usa t_obs no negocio la de fazer os dado
t_start(i) = indexBp.date_d$obs
t_stop(i) = indexBp2.date_d$obs
dt(i) = anytim(t_stop(i))-anytim(t_start(i))


ENDFOR

;alterar path
pathout_save='/usr/users/UNN/w17016451/saves/'

save, Bp_start, Bp_stop, Br_start, Br_stop, Bt_start, Bt_stop, t_start, t_stop, dt, filename=strjoin(pathout_save+strjoin('dave4vmparameters.save'))


end
