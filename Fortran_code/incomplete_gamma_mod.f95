module incomplete_gamma_mod
  implicit none
  integer, parameter :: max_iter = 200
  real(8), parameter :: eps = epsilon(1.0_8)
contains

function upper_incomplete_gamma_function(s, x) result(upper_gamma)
    real(8), intent(in) :: s, x
    real(8) :: upper_gamma
    real(8) :: d, c, f, delta, a, b, log_prefix
    integer :: j

    if (x <= 0.0_8) then
        upper_gamma = 1.0_8
        return
    end if

    log_prefix = s * log(x) - x - log_gamma(s)

    f = x
    if (abs(f) < eps) f = eps
    c = f
    d = 0.0_8

    do j = 1, max_iter
        a = -real(j, 8) * (real(j, 8) - s)
        b = 2.0_8 * j + 1.0_8 - s + x
        
        d = b + a * d
        if (abs(d) < eps) d = eps
        c = b + a / c
        if (abs(c) < eps) c = eps
        
        d = 1.0_8 / d
        delta = c * d
        f = f * delta
        
        if (abs(delta - 1.0_8) < eps) exit
    end do

    upper_gamma = exp(log_prefix) / f
end function upper_incomplete_gamma_function

function lower_incomplete_gamma_function(s, x) result(lower_gamma)
    real(8), intent(in) :: s, x
    real(8) :: lower_gamma
    real(8) :: term, sum_val, log_prefix
    integer :: i

    if (x <= 0.0_8) then
        lower_gamma = 0.0_8
        return
    end if

    log_prefix = s * log(x) - x - log_gamma(s)

    term = 1.0_8 / s
    sum_val = term

    do i = 1, max_iter
        term = term * (x / (s + real(i, 8)))
        sum_val = sum_val + term
        if (abs(term) < abs(sum_val) * eps) exit
    end do

    lower_gamma = sum_val * exp(log_prefix)

end function lower_incomplete_gamma_function

end module incomplete_gamma_mod