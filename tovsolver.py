#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d


# From TOV's equation


def calc_dm(rho_, r_, dr_):
    return 4. * np.pi * r_ * r_ * rho_ * dr_


def calc_dp(rho_, p_, r_, dr_, m_):
    return - G*m_*rho_/r_/r_*(1.+p_/rho_/c/c) * \
        (1.+4.*np.pi*r_*r_*r_*p_/m_/c/c) / \
        (1.-2.*G*m_/r_/c/c) * dr_


# Basic method of integration x3 for the MoI


def calc_dphi(rho_, p_, dp_):
    return - dp_/rho_/c/c/(1+p_/rho_/c/c)


def calc_dj(r_, rho_, p_, phi_, m_, wbar_, dr_):
    return 8.*np.pi*r_*r_*r_*r_/3. * (rho_+p_/c/c) * \
        np.exp(-phi_) * np.sqrt(1.-2.*G*m_/r_/c/c) * wbar_ * dr_


def calc_dwbar(phi_, m_, r_, j_, dr_):
    return G * np.exp(phi_) * j_/c/c/r_/r_/r_/r_ / \
        np.sqrt(1.-2.*G*m_/r_/c/c) * dr_


def calc_OMEGA(wbar_R_, J_, R_):
    return wbar_R_ + 2.*G*J_/c/c/R_/R_/R_


def calc_dI(r_, rho_, p_, wbar_, OMEGA_, m_, phi_, dr_):
    return 8.*np.pi*r_*r_*r_*r_/3. * (rho_+p_/c/c) * wbar_ / OMEGA_ * \
        np.sqrt(1.-2.*G*m_/r_/c/c) * np.exp(-phi_) * dr_


# Simpler exotic method from Lattimer and Prakash's paper


def calc_dw(rho_, p_, w_, r_, m_, dr_):
    return 4.*np.pi*G * (rho_*c*c+p_) * (4.+w_) * r_ * dr_ / \
        (c*c*(c*c-2.*G*m_/r_)) - w_/r_ * (3+w_) * dr_


def calc_I_LP(wR_, R_):
    return c*c*wR_*R_*R_*R_ / (G*(6.+2.*wR_))


# Approximation from Ravenhall and Pethick for a 0.8-1.6 M0 NS (1993)


def calc_RP_approx_I(M_, R_):
    return 0.21*M_*R_*R_ / (1.-2.*G*M_/R_/c/c)


# Better approximation from Lattimer & Schutz that permits greater
# masses than 1.6 M0 (2006) [M0.km^{2}]


def calc_LS_approx_I(M_, R_):
    return 0.237*M_*R_*R_ * (1.+2.84*G*M_/c/c/R_ +
                             18.9 * (G*M_/c/c/R_)**4.)


# Classical MoI


def calc_dI_clas(r_, rho_, dr_):
    return 8. * np.pi / 3. * r_ * r_ * r_ * r_ * rho_ * dr_


# Computing of the core radius


def calc_Rcore(mucc_, mu0_, R_, M_):
    alpha = (mucc_ / mu0_) * (mucc_ / mu0_)
    rg = 2. * G * M_ / c / c
    return alpha*R_ / (1.+R_*(alpha-1.)/rg)


if __name__ == "__main__":

    # Physical constants in CGS units
    G = 6.6742e-8
    c = 2.99792458e10
    msun = 1.989e33
    p_conv_fact = 1.6022e33         # [MeV/fm^{3}] to [dyn/cm^{2}]
    I_conv_fact = 1e-10 / 1.989e33  # [g.cm^{2}] to [M0.km^{2}]
    # p_conv_fact = 1. # if pressure is already given in [dyn/cm^{2}]

    # Interpolation of the mass density and the pressure
    # for a table | nB | rhoB | p |
    eos = "eos/sly4d_eos.data"
    with open(eos, "rb") as eos_table:
        density_col = eos_table.read().split()[1::3]
    with open(eos, "rb") as eos_table:
        pressure_col = eos_table.read().split()[2::3]
    for i in range(len(density_col)):
        density_col[i] = float(density_col[i])
        pressure_col[i] = float(pressure_col[i]) * p_conv_fact
    pressure_interp = interp1d(density_col, pressure_col,
                               fill_value="extrapolate")
    density_interp = interp1d(pressure_col, density_col,
                              fill_value="extrapolate")

    def calc_pressure(rho_):
        return pressure_interp(rho_)

    def calc_density(p_):
        return density_interp(p_)

    # Integration parameters
    dr = 1000.        # Step [cm]
    rho_sat = 2.3e14  # Nuclear saturation density [g/cm^3]
    rho_max = 1.e15   # Alternative to avoid the extrapolation above 7*rho_sat
    N = 50            # Number of points (grid)
    # Minimal density [g/cm^3]; very reasonable choice for a NS
    rho_min = 2.e5

    # Output
    MofR = open("out/data_sly4d.data", "w")   # File for M(R) plot
    # File for density/mass/pressure profiles a NS given the model
    profiles = open("out/profile_sly4d.data", "w")
    MoI = open("out/MoI_sly4d.data", "w")     # File for I(M)

    # Routine
    for i in range(0, N):

        list_r = []
        list_m = []
        list_p = []
        list_rho = []
        list_phi = []
        list_dp = []
        list_dr = []
        list_wbar = []
        list_j = []

        # Initialization
        rho_c = rho_sat + i * (9. * rho_sat) / N  # Central density
        # rho_c = rho_sat + i*(rho_max-rho_sat)/N # Alternative interval for bad extrapolations
        rho = rho_c
        m = 0.                     # m(r=0) = 0
        r = 10.                    # To avoid the singularity at r=0
        pc = calc_pressure(rho_c)  # Central pressure
        p = pc
        w = 0.1                    # Boundary limit for the Lat&Prak function at r=0
        I_clas = 0.                # Iclas(r=0)

        # Profile for a single NS from r=0 to r=R
        while rho > rho_min:
            r = r + dr
            dm = calc_dm(rho, r, dr)
            m = m + dm
            dp = calc_dp(rho, p, r, dr, m)
            dw = calc_dw(rho, p, w, r, m, dr)
            dI_clas = calc_dI_clas(r, rho, dr)
            p = p + dp
            w = w + dw
            I_clas = I_clas + dI_clas
            if p > 1e20:
                rho = float(calc_density(p))
            else:
                break

            list_r.append(r)
            list_m.append(m)
            list_p.append(p)
            list_rho.append(rho)
            list_dp.append(dp)
            list_dr.append(dr)

            profiles.write(
                "{} {} {} {}\n".format(r/100000, m/msun, rho, p))

        R = r / 100000  # Radius [km]
        M = m / msun    # Mass [M_sun]

        # Total MoI with Lattimer and Prakash's model [M0.km^{2}]
        I_LP = calc_I_LP(w, r) * I_conv_fact

        # Approximated MoI for a 0.8-1.6 M0 NS [M0.km^{2}]
        I_RP_approx = calc_RP_approx_I(m, r) * I_conv_fact

        # Better approximation allowing greater mass [M0.km{2}]
        I_LS_approx = calc_LS_approx_I(m, r) * I_conv_fact

        MofR.write("{} {} {} {}\n".format(R, M, rho_c, pc))
        MoI.write("{} {} {} {} {} ".format(
            M, I_RP_approx, I_LS_approx, I_LP, I_clas * I_conv_fact))

        # Second initialization for opposite integration
        phi = np.log(np.sqrt(1 - 2 * G * m / c / c / r))

        # Way back from r=R to r=0 to use the known condition for phi(r=R)
        for i in range(len(list_p)):
            dphi = calc_dphi(list_rho[- 1 - i],
                             list_p[- 1 - i], list_dp[- 1 - i])
            phi = phi - dphi

            list_phi.append(phi)

        # Initialization for the 3rd integration
        # Arbitrary input for w(r=0) little enough to fit with the reality
        wbar = 1.
        j = 0.  # Obviously j(r=0) = 0 (j = Angular momentum for a sphere of radius r)

        # From 0 to R to get the MoI with the common method
        for i in range(len(list_r)):
            dj = calc_dj(list_r[0 + i], list_rho[0 + i], list_p[0 + i], list_phi[-1 - i],
                         list_m[0 + i], wbar, list_dr[0 + i])
            dwbar = calc_dwbar(list_phi[-1 - i], list_m[0 + i],
                               list_r[0 + i], j, list_dr[0 + i])
            j = j + dj
            wbar = wbar + dwbar

            list_j.append(j)
            list_wbar.append(wbar)

        # Final computing of the MoI
        OMEGA = calc_OMEGA(wbar, j, r)
        I = j / OMEGA * I_conv_fact
        # print(I)
        # I      = 0. # I(r=0)
        # for b in range(len(list_r)):
        #     dI = calc_dI(list_r[0 + b], list_rho[0 + b], list_p[0 + b], list_wbar[0 + b],
        #                  OMEGA, list_m[0 + b], list_phi[- 1 - b], list_dr[0 + b])
        #     I  = I + dI

        MoI.write("{}\n".format(I))

    MofR.close()
    profiles.close()
    MoI.close()
