program fldcurv
    use magnetic_model
    use incomplete_gamma_mod
    implicit none

    real(8) :: value, t1, t2, s, x
    integer :: i

    s = 3.0_8
    x = 7.0_8

    call cpu_time(t1)

    do i = 1, 1000000
        value = upper_incomplete_gamma_function(s, x) + lower_incomplete_gamma_function(s, x)
    end do

    call cpu_time(t2)

    write(*, *) 'value', value, 'time', (t2 - t1) / 1000000

    write(*, *) 'upper', upper_incomplete_gamma_function(s, x) * gamma(s)
    write(*, *) 'lower', lower_incomplete_gamma_function(s, x) * gamma(s)

    write(*, *) abs(upper_incomplete_gamma_function(s, x) * gamma(s) + &
    & lower_incomplete_gamma_function(s, x) * gamma(s) - gamma(s)) / gamma(s)

end program fldcurv
