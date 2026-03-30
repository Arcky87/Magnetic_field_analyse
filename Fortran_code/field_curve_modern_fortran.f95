program fldcurv
    use magnetic_model
    implicit none

    integer :: i, j, k, num_observ, num_phases, num_i, num_beta, num_bp0, loc_max(3)
    real(8) :: t1, t2
    real(8), allocatable, dimension(:) :: observ_magnetic_field, observ_err_magnetic_field
    real(8), allocatable, dimension(:) :: phase_vector, i_vector, beta_vector, bp0_vector
    real(8), allocatable, dimension(:, :, :) :: posterior_map

    open(15, file='fortran_data.dat', status='old', action='read')
    open(16, file='fortran_maps_output.dat', status='old', action='write')

    read(15, *) num_observ

    num_phases = 72
    num_i = 36
    num_beta = 36
    num_bp0 = 250

    allocate(phase_vector(1:num_phases), i_vector(1:num_i), beta_vector(1:num_beta), bp0_vector(1:num_bp0))
    allocate(observ_magnetic_field(1:num_observ), observ_err_magnetic_field(1:num_observ))
    allocate(posterior_map(1:num_beta, 1:num_i, 1:num_bp0))

    do i = 1, num_observ
        read(15, *) observ_magnetic_field(i), observ_err_magnetic_field(i)
    end do

    phase_vector = (/( i * (1.0_8 - 0.0_8) / real(num_phases - 1, 8), i = 0, num_phases - 1 )/)
    i_vector = (/( i * (pi - 0.0_8) / real(num_i - 1, 8), i = 0, num_i - 1 )/)
    beta_vector = (/( i * (pi - 0.0_8) / real(num_beta - 1, 8), i = 0, num_beta - 1 )/)
    bp0_vector = (/( i * (10000.0_8 - 0.0_8) / real(num_bp0 - 1, 8), i = 0, num_bp0 - 1 )/)

    call cpu_time(t1)

    call posterior_result(observ_magnetic_field, observ_err_magnetic_field, i_vector, beta_vector, & 
    & bp0_vector, phase_vector, posterior_map)

    call cpu_time(t2)

    write(*, *) 'Time compute:', (t2 - t1) / 16, 'c'

    do i = 1, num_beta
        do j = 1, num_i
            write(16, *) posterior_map(i, j, :)
        end do
    end do

    loc_max = maxloc(posterior_map)

    write(*, *) beta_vector(loc_max(1)) * 180.0_8 / pi, i_vector(loc_max(2)) * 180.0_8 / pi, bp0_vector(loc_max(3))

    write(*, *) sum(sum(sum(posterior_map, dim=1), dim=1) * bp0_vector)

    deallocate(observ_magnetic_field, observ_err_magnetic_field)
    deallocate(phase_vector, i_vector, beta_vector, bp0_vector)
    deallocate(posterior_map)

end program fldcurv
