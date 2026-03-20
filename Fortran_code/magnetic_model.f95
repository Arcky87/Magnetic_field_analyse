module magnetic_model
    implicit none

    real(8), parameter :: pi = 3.141592653589793_8

contains

function prior_polar_magnetic_field(polar_field) result(distribution)
    implicit none
        real(8), intent(in) :: polar_field
        real(8) :: distribution
        real(8) :: a, polar_magnetic_field_max

        a = 25.0_8
        polar_magnetic_field_max = 1.0_8 * 1.0E+4

        distribution = 1.0_8 / (polar_field + a) / log((a + polar_magnetic_field_max) / a)

    end function prior_polar_magnetic_field


    function prior_declination_of_rotation(declination_of_rotation) result(distribution)
    implicit none
        real(8), intent(in) :: declination_of_rotation
        real(8) :: distribution

        distribution = 0.5_8 * sin(declination_of_rotation)

    end function prior_declination_of_rotation


    function prior_declination_of_magnetic_field(declination_of_magnetic_field) result(distribution)
    implicit none
        real(8), intent(in) :: declination_of_magnetic_field
        real(8) :: distribution
        real(8) :: declination_of_magnetic_field_max, declination_of_magnetic_field_min

        declination_of_magnetic_field_min = 0.0_8
        declination_of_magnetic_field_max = pi

        distribution = 1.0_8 / (declination_of_magnetic_field_max - declination_of_magnetic_field_min)

    end function prior_declination_of_magnetic_field


    function prior_phase(phase) result(distribution)
    implicit none
        real(8), intent(in) :: phase
        real(8) :: distribution
        real(8) :: phase_max, phase_min

        phase_min = 0.0_8
        phase_max = 1.0_8

        distribution = 1.0_8 / (phase_max - phase_min)

    end function prior_phase


    function prior_scale_coef(b) result(distribution)
    implicit none
        real(8), intent(in) :: b
        real(8) :: distribution
        real(8) :: b_min, b_max

        b_min = 0.1_8
        b_max = 2.0_8

        distribution = 1.0_8 / b / log(b_max / b_min)

    end function prior_scale_coef


    function likelihood_mod(observe_data, observe_err, model_data, phases, b) result(likelihood)
    implicit none
        real(8), intent(in), dimension(:) :: observe_data, observe_err, model_data, phases 
        real(8), intent(in) :: b
        real(8) :: likelihood
        real(8) :: chi2
        real(8), dimension(size(observe_data)) :: likelihood_observ_data
        real(8), dimension(size(model_data)) :: likelihood_model_data
        integer :: i, k, num_model_data, num_observ

        num_model_data = size(model_data)
        num_observ = size(observe_data)

        do i = 1, num_observ
            do k = 1, num_model_data
                likelihood_model_data(k) = exp(-b / 2.0_8 * ((model_data(k) - observe_data(i)) / observe_err(i)) ** 2.0_8) &
                &* prior_phase(phases(k))
            end do

            chi2 = sum(likelihood_model_data)

            likelihood_observ_data(i) = 1.0_8 / observe_err(i) * chi2

        end do

        likelihood = product(likelihood_observ_data) * (sqrt(b / 2.0_8 / pi) / real(num_model_data, 8)) ** real(num_observ, 8)

    end function likelihood_mod

    subroutine prior_result_by_value(i_angle, beta, bp0, b, prior_distribution)

    implicit none
        real(8), intent(in) :: i_angle, beta, bp0, b
        real(8), intent(out) :: prior_distribution

            prior_distribution = prior_declination_of_magnetic_field(beta) * &
            & prior_declination_of_rotation(i_angle) * prior_polar_magnetic_field(bp0) * prior_scale_coef(b)

    end subroutine prior_result_by_value

    subroutine posterior_by_value(observe_data, observe_data_err, decline_rotation, decline_magnetic_field, & 
    & polar_field, scale_factor, phases, posterior_distribution)

    implicit none
        real(8), intent(in) :: decline_rotation, decline_magnetic_field, &
        & polar_field, scale_factor
        real(8), intent(in), dimension(:) :: phases, observe_data, observe_data_err
        real(8), intent(out) :: posterior_distribution
        real(8) :: likelihood, prior_distribution
        real(8), dimension(size(phases)) :: model_data
        integer :: i, num_observ_data, num_model_data

        num_observ_data = size(observe_data)
        num_model_data = size(phases)

        call prior_result_by_value(decline_rotation, decline_magnetic_field, polar_field, & 
        &  scale_factor, prior_distribution)

        do i = 1, num_model_data
            call compute_bl(phases(i), decline_rotation * 180.0_8 / pi, decline_magnetic_field * 180.0_8 / pi, polar_field, &
            &0.0_8, 0.0_8, 0.0_8, 0.35_8, 0.25_8, model_data(i))
        end do

        likelihood = likelihood_mod(observe_data, observe_data_err, model_data, phases, scale_factor)

        posterior_distribution = prior_distribution * likelihood 

    end subroutine posterior_by_value

    subroutine posterior_result(observe_data, observe_data_err, declines_rotation, declines_magnetic_field, & 
    & polar_fields, scale_factors, phases, posterior_distribution)

        real(8), intent(in), dimension(:) :: observe_data, observe_data_err, declines_rotation, declines_magnetic_field, & 
        & polar_fields, scale_factors, phases
        real(8), intent(out), dimension(size(declines_magnetic_field), size(declines_rotation), size(polar_fields)& 
        &) :: posterior_distribution
        real(8), dimension(size(declines_magnetic_field), size(declines_rotation), size(polar_fields), & 
        & size(scale_factors)) :: posterior_distribution_pre
        integer :: num_beta, num_i, num_bp0, num_b
        integer :: j, k, l, m
        real(8) :: evidence, beta, i_angle, bp0, b
        real(8) :: dbeta, di, dbp0, db, t1

        num_beta = size(declines_magnetic_field)
        num_i = size(declines_rotation)
        num_bp0 = size(polar_fields)
        num_b = size(scale_factors)

        dbeta = abs(declines_magnetic_field(2) - declines_magnetic_field(1))
        di = abs(declines_rotation(2) - declines_rotation(1))
        dbp0 = abs(polar_fields(2) - polar_fields(1))
        db = abs(scale_factors(2) - scale_factors(1))

        do j = 1, num_beta
            call cpu_time(t1)
            write(*, *) 'Present num ', j, 'Max ', num_beta, 'time mark', t1
            do k = 1, num_i
                do l = 1, num_bp0
                    do m = 1, num_b
                        beta = declines_magnetic_field(j) 
                        i_angle = declines_rotation(k) 
                        bp0 = polar_fields(l)
                        b = scale_factors(m)

                        call posterior_by_value(observe_data, observe_data_err, i_angle, beta, & 
                        & bp0, b, phases, posterior_distribution_pre(j, k, l, m))
                    end do
                end do
            end do
        end do

        evidence = sum(posterior_distribution_pre) * dbeta * di * db * dbp0

        posterior_distribution_pre = posterior_distribution_pre / evidence

        posterior_distribution = sum(posterior_distribution_pre, dim=4) * db

    end subroutine posterior_result

    subroutine compute_bl(phase, ai, beta, bp0, ad, bq, boct, a_ld, b_ld, bl)

    implicit none

        real(8), intent(in) :: phase
        real(8), intent(in) :: ai, beta
        real(8), intent(in) :: bp0, ad, bq, boct
        real(8), intent(in) :: a_ld, b_ld
        real(8), intent(out) :: bl

        real(8) :: bs

        call disk_field(ai, beta, phase, bp0, ad, bq, boct, a_ld, b_ld, bl, bs)

    end subroutine compute_bl

    subroutine disk_field(ai_deg,beta_deg,phase,bp0,ad,bq,boct,a_ld,b_ld,bl,bs)

        implicit none

        real(8), intent(in) :: ai_deg,beta_deg
        real(8), intent(in) :: phase
        real(8), intent(in) :: bp0,ad,bq,boct
        real(8), intent(in) :: a_ld,b_ld

        real(8), intent(out) :: bl,bs

        integer :: i, j, itot, ju
        real(8) :: ai, beta, phi
        real(8) :: x, y, z, r, th
        real(8) :: area
        real(8) :: bx, by, bz, b
        real(8) :: mu
        real(8) :: weight
        real(8) :: sum_w

        real(8) :: cos_th, sin_th

        itot = 40

        ai = ai_deg*pi/180.0_8
        beta = beta_deg*pi/180.0_8
        phi = 2.0_8*pi*phase

        bl = 0.0_8
        bs = 0.0_8
        sum_w = 0.0_8

        do i = 1,itot

            r = (real(i,8)-0.5_8)/real(itot,8)

            area = pi*(2.0_8*i-1.0_8)/(6.0_8*i*(real(itot,8)**2))

            ju = 6*i

            do j = 1,ju

                th = 2.0_8*pi*(real(j,8)-0.5_8)/real(ju,8)

                if (mod(i,2) == 0) then
                    th = th + pi/real(ju,8)
                end if

                cos_th = cos(th)
                sin_th = sin(th)

                x = r*cos_th
                y = r*sin_th
                z = sqrt(1.0_8 - x*x - y*y)

                call magnetic_field(x,y,z,ai,beta,phi,bp0,ad,bq,boct,bx,by,bz)

                b = sqrt(bx*bx + by*by + bz*bz)

                mu = z

                weight = (1.0_8 - a_ld*(1.0_8-mu) - b_ld*(1.0_8-mu)**2) * area

                sum_w = sum_w + weight
                bl = bl + bz*weight
                bs = bs + b*weight

            end do
        end do

        if (sum_w > 0.0_8) then
            bl = bl / sum_w
            bs = bs / sum_w
        end if

    end subroutine disk_field



    subroutine magnetic_field(x,y,z,ai,beta,phi,bp0,ad,bq,boct,bx,by,bz)

        implicit none

        real(8), intent(in) :: x,y,z
        real(8), intent(in) :: ai,beta,phi
        real(8), intent(in) :: bp0,ad,bq,boct

        real(8), intent(out) :: bx,by,bz

        real(8) :: xr,yr,zr
        real(8) :: xm,ym,zm

        real(8) :: bxm,bym,bzm
        real(8) :: bxr,byr,bzr

        real(8) :: cai,sai,cb,sb,cp,sp
        real(8) :: d2,d5

        cai = cos(ai)
        sai = sin(ai)
        cb = cos(beta)
        sb = sin(beta)
        cp = cos(phi)
        sp = sin(phi)

        xr = x*cai - z*sai
        yr = y
        zr = x*sai + z*cai

        xm = (xr*cp + yr*sp)*cb + zr*sb
        ym = -xr*sp + yr*cp
        zm = -(xr*cp + yr*sp)*sb + zr*cb

        bxm = 0.0_8
        bym = 0.0_8
        bzm = 0.0_8

        if (abs(bp0) > 0.1_8) then

            d2 = 1.0_8 + ad*ad - 2.0_8*ad*zm
            d5 = d2**2.5_8

            bxm = bxm + bp0*3.0_8*xm*(zm-ad)/(2.0_8*d5)
            bym = bym + bp0*3.0_8*ym*(zm-ad)/(2.0_8*d5)
            bzm = bzm + bp0*(3.0_8*(zm-ad)**2 - d2)/(2.0_8*d5)

        end if

        if (abs(bq) > 0.1_8) then

            bxm = bxm + bq*xm*(5.0_8*zm**2 - 1.0_8)/2.0_8
            bym = bym + bq*ym*(5.0_8*zm**2 - 1.0_8)/2.0_8
            bzm = bzm + bq*zm*(5.0_8*zm**2 - 3.0_8)/2.0_8

        end if

        if (abs(boct) > 0.1_8) then

            bxm = bxm + boct*5.0_8*(7.0_8*zm**3 - 3.0_8*zm)*xm/8.0_8
            bym = bym + boct*5.0_8*(7.0_8*zm**3 - 3.0_8*zm)*ym/8.0_8
            bzm = bzm + boct*(35.0_8*zm**4 - 30.0_8*zm**2 + 3.0_8)/8.0_8

        end if

        bxr = (bxm*cb - bzm*sb)*cp - bym*sp
        byr = (bxm*cb - bzm*sb)*sp + bym*cp
        bzr = bxm*sb + bzm*cb

        bx = bxr*cai + bzr*sai
        by = byr
        bz = -bxr*sai + bzr*cai

    end subroutine magnetic_field

end module magnetic_model