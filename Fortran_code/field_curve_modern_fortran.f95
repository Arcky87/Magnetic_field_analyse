program fldcurv
    use magnetic_model
    implicit none

    real(8) :: a_ld, b_ld, ai, beta, t1, t2
    real(8) :: bp0, ad, bq, boct
    real(8) :: phase, bl, bs, step_phase
    integer :: i, num_of_phase

    integer :: num_beta, num_i, num_bp, num_b, num_phi
    real(8) :: step_beta, step_i, step_bp, step_b, step_phi

    real(8) :: phi_vector(72), beta_vector(36), i_vector(37), b_vector(40)
    real(8) :: bp_vector(250), prior_distribution(72, 36, 37, 250, 40)

    ! open(unit=15,file="fldcurv.dat",status="old")
    ! open(unit=16,file="fldcurv_f95.out",status="replace")
    ! open(unit=12,file="be_f95.out",status="replace")
    ! open(unit=13,file="bs_f95.out",status="replace")

    ! read(15,*) a_ld,  b_ld, ai, beta, bp0, ad, bq, boct

    ! num_of_phase = 20
    ! step_phase = 1.0_8 / num_of_phase

    ! do i = 0, num_of_phase

    !     phase = step_phase * real(i, 8)

    !     call disk_field(ai, beta, phase, bp0, ad, bq, boct, a_ld,  b_ld, bl, bs)

    !     write(16,'(F8.2,2F12.3)') phase, bl, bs
    !     write(12,'(F12.3)') bl
    !     write(13,'(F12.3)') bs

    ! end do

    write(*, *) prior_polar_magnetic_field(300.0_8)

end program fldcurv
