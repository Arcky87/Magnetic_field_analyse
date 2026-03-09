module magnetic_model
    implicit none

    real(8), parameter :: pi = 3.141592653589793_8

contains

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

    integer :: i,j,itot,ju
    real(8) :: ai,beta,phi
    real(8) :: x,y,z,r,th
    real(8) :: area
    real(8) :: bx,by,bz,b
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