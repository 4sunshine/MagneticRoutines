## Status


## Questions  

### TMF  
1. I wrote the data in the following way, is it also correct?  

```IDL
Bvec[*, *, cur_time_id, 0] = data.BX  ;; data.BX is 2D-Array of (nx, ny) shape
Bvec[*, *, cur_time_id, 1] = data.BY
Bvec[*, *, cur_time_id, 2] = data.BZ

Vvec[*, *, cur_time_id, 0] = data.VX
Vvec[*, *, cur_time_id, 1] = data.VY
Vvec[*, *, cur_time_id, 2] = data.VZ
```

2. How do you handle cases when there is no HMI data ~ 6-8 UTC?  

3. Can I start TMF from Non-linear FFF extrapolation?  A: Yes.

4. I didn't added ghost cells for boundary conditions, is it correct?    

5. Why Momentum sets not with:
```Fortran
subroutine driven_velocity(ixI^L, ixO^L, qt, w)
  use mod_global_parameters
  integer, intent(in) :: ixI^L, ixO^L
  double precision, intent(in) :: qt
  double precision, intent(inout) :: w(ixI^S, nw)

  ! Set inverted velocity in ghost cells
  w(ixO^S, mom(1)) = v_inv_x(qt, x, y) * w(ixO^S, rho_)
  w(ixO^S, mom(2)) = v_inv_y(qt, x, y) * w(ixO^S, rho_)
end subroutine driven_velocity
```

but with  
```Fortran
w(ix^D,mom(1):mom(3))=(Vddb(ixbc1,ixbc2,ixv,1:ndir) + &
    (Vddb(ixbc1,ixbc2,ixv+1,1:ndir) - Vddb(ixbc1,ixbc2,ixv,1:ndir))/dtbv* &
    (qt-t_driven_v(ixv)))
```

### MHD  

### General  

1. Can I create SDO-AIA Images with MPI-AMRVAC?
