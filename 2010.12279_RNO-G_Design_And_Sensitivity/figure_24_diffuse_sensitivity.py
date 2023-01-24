from NuRadioMC.EvtGen.generator import ice_cube_nu_fit
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import erf, gamma, erfinv
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limit
from NuRadioMC.utilities.cross_sections import get_interaction_length
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

def total_events_ice_cube(slope, energy_centres, effective_volumes,
                          livetime=2/3*5*units.year):

    log_energy_centres = np.log10(energy_centres)
    log_delta_energy = log_energy_centres[1] - log_energy_centres[0]
    log_left_bins = log_energy_centres - log_delta_energy/2
    log_right_bins = log_energy_centres + log_delta_energy/2

    left_bins = 10 ** log_left_bins
    right_bins = 10 ** log_right_bins

    number_events = 0

    for left_bin, right_bin, energy_centre, effective_volume in zip(left_bins, right_bins, energy_centres, effective_volumes):

        integrated_flux = quad(ice_cube_nu_fit, left_bin, right_bin, args=(slope))[0]

        effective_area = effective_volume / get_interaction_length(energy_centre)
        solid_angle = 4 * np.pi * units.sr
        number_events_bin = integrated_flux * effective_area * livetime * solid_angle

        number_events += number_events_bin

    return number_events

def get_excluded_slope(energy_centres, effective_volumes,
                       min_energy=20*units.PeV):

    mask_energy = energy_centres > min_energy

    upper_lim_events = 2.44

    def events_root(slope, energy_centres_masked, effective_volumes_masked):

        return total_events_ice_cube(slope, energy_centres_masked, effective_volumes_masked) - upper_lim_events

    excluded_slope = fsolve(events_root, -2.1,
                            args=(energy_centres[mask_energy], effective_volumes[mask_energy]))[0]

    return excluded_slope


def get_E2_upper_limits(energy_centres,
                        effective_volumes,
                        upper_nevt=2.44,
                        independent_stations=1,
                        livetime=1*units.year,
                        solid_angle=4*np.pi):
    """
    This function returns an upper limit flux multiplied by the square of the energy.

    Parameters
    ----------
    energy_centres: array of floats
        Bin centre energy
    effective_volumes: array of floats
        Effective volumes
    upper_nevt: integer
        The number of events that correspond to the desired confidence level.
        For an experiment detecting no events, the 90% CL limit for the mean
        number of events is 2.44.
    independent_stations: integer
        If the effective volume file contains only one simulated station, the
        result can be multiplied by this number to get an estimate for the whole
        array. Careful! This only works if stations are independent, and they
        are not for high energies, most of the time.
    livetime: float
        Time the array is expected to take data
    solid_angle: float
        Solid angle to multiply the effective volume. By default we consider the
        whole sky (4 pi)

    Returns
    -------
    upper_flux_E2: array of floats
        Differential pper limit flux times neutrino energy squared
    """

    log_energy_centres = np.log10(energy_centres)
    log_delta_energy = log_energy_centres[1] - log_energy_centres[0]
    log_left_bins = log_energy_centres - log_delta_energy/2
    log_right_bins = log_energy_centres + log_delta_energy/2

    left_bins = 10 ** log_left_bins
    right_bins = 10 ** log_right_bins

    effective_areas = effective_volumes / get_interaction_length(energy_centres, cross_section_type='ctw')
    effective_areas *= independent_stations

    upper_flux = upper_nevt / ( effective_areas * solid_angle * livetime * (right_bins-left_bins) )
    upper_flux_E2 = upper_flux * energy_centres**2

    return upper_flux_E2

def get_flux_uncertainty(effective_volumes, effective_volumes_uncertainty, upper_limits):
    """
    This function calculates the propagation of uncertainty from the effective volumes
    to the upper limits using a linearised expansion for the propagation. A better way
    would be to run a Monte Carlo.
    """

    upper_limits_uncertainty = upper_limits * ( effective_volumes_uncertainty / effective_volumes )

    return upper_limits_uncertainty

def gaussian_inverse_CDF(F_2tailed):

    # Two-tailed gaussian inverse CDF
    # Using Z = scipy.stats.norm.chi2.ppf(F_2tailed, 1) would work as well.
    # The following implementation is kept for pedagogical purposes.
    from scipy.stats import norm
    area_tail = (1 - F_2tailed)/2
    F_1tailed = F_2tailed + area_tail
    Z = norm.ppf(F_1tailed)

    return Z

def get_E2_upper_limits_asimov(energy_centres,
                               effective_volumes,
                               independent_stations=1,
                               livetime=1*units.year,
                               solid_angle=4*np.pi):

    log_energy_centres = np.log10(energy_centres)
    log_delta_energy = log_energy_centres[1] - log_energy_centres[0]
    log_left_bins = log_energy_centres - log_delta_energy/2
    log_right_bins = log_energy_centres + log_delta_energy/2

    left_bins = 10 ** log_left_bins
    right_bins = 10 ** log_right_bins

    effective_areas = effective_volumes / get_interaction_length(energy_centres, cross_section_type='ctw')
    effective_areas *= independent_stations

    background_fluxes = np.zeros_like(energy_centres)

    def number_of_events(left_bin, right_bin, flux, effective_area):

        return effective_area * solid_angle * livetime * (right_bin-left_bin) * flux * independent_stations

    def log_likelihood_mu(n_s, n_b, n_t):

        if (n_s + n_b) < 1e-50:

            n_s = 1e-50
            n_b = 1e-50

        #log_L = n_t * np.log(n_s + n_b)
        #log_L -= n_s + n_b
        #log_L -= np.log(gamma(n_t + 1))

        #return log_L

        #print(n_s, n_b, n_t)
        L = (n_s + n_b) ** (n_t) / gamma(n_t + 1) * np.exp(-(n_s + n_b))

        return np.log(L)

    def log_likelihood_ratio(n_s, n_b, n_t):

        ratio  = log_likelihood_mu(n_s, n_b, n_t)
        ratio -= log_likelihood_mu(0, n_b, n_t)

        return ratio

    def q_mu(n_signal, n_background, n_total_avg):

        print(-2 * log_likelihood_ratio(n_signal, n_background, n_total_avg), gaussian_inverse_CDF(0.9)**2)
        return -2 * log_likelihood_ratio(n_signal, n_background, n_total_avg)

    def Z_mu_root(signal_flux, left_bin, right_bin, background_flux, effective_area):

        print('signal', signal_flux)
        print('cumu', q_mu(left_bin, right_bin, signal_flux, background_flux, effective_area))
        CL = 0.9 # 90% CL
        return np.sqrt(q_mu(left_bin, right_bin, signal_flux, background_flux, effective_area)) - erfinv(CL)

    def Z_mu_root_events(n_signal, n_background, n_total_avg):

        CL = 0.9 # 90% CL
        return np.sqrt(q_mu(n_signal, n_background, n_total_avg)) - gaussian_inverse_CDF(0.9)

    upper_limits = []

    for left_bin, right_bin, effective_area, background_flux in zip(left_bins,
                                                                      right_bins,
                                                                      effective_areas,
                                                                      background_fluxes):

        n_background_avg = number_of_events(left_bin, right_bin, background_flux, effective_area) + 1e-20
        n_total_avg = n_background_avg

        upper_limit = fsolve(Z_mu_root_events, 2.4,
                             args=(n_background_avg + 0., n_total_avg + 0.))[0]
        print('waaa', Z_mu_root_events(1.3, 1e-20, 1e-20))

        upper_limits.append(upper_limit)

        #print('vamoh a ahsh')
        #n_ev=number_of_events(left_bin, right_bin, 1e-45 * units.GeV**-1 * units.cm**-2 * units.s**-1 * units.sr**-1, effective_area)
        #print('printear', n_ev)
        #print("miuuuuuu", q_mu(left_bin, right_bin, 1e-45 * units.GeV**-1 * units.cm**-2 * units.s**-1 * units.sr**-1, background_flux, effective_area))

    print(upper_limits)
    exit()
    upper_limits_E2 = np.array(upper_limits) * energy_centres ** 2

    return upper_limits_E2

def get_flux_unc_MC(energy_centres,
                    effective_volumes,
                    effective_volumes_uncertainty,
                    upper_nevt=2.44,
                    independent_stations=1,
                    livetime=1*units.year,
                    solid_angle=4*np.pi,
                    Ntries=10000,
                    calculate_slopes=False):

    upper_limits = []
    mode = 'gauss'

    stat_method = 'FC' # 'FC' or 'Asimov'

    excluded_slopes = []

    for Ntry in range(Ntries):

        if mode == 'gauss':
            effective_volumes_draw = np.random.normal(effective_volumes, effective_volumes_uncertainty)
            effective_volumes_draw[effective_volumes_draw < 0] = 1 * units.m3
        elif mode == 'poisson':
            pass

        if stat_method == 'FC':
            upper_limits_draw = get_E2_upper_limits(energy_centres, effective_volumes_draw,
                                                    upper_nevt=upper_nevt,
                                                    independent_stations=independent_stations,
                                                    livetime=livetime,
                                                    solid_angle=solid_angle)
        elif stat_method == 'Asimov':

            upper_limits_draw = get_E2_upper_limits_asimov(energy_centres, effective_volumes_draw,
                                                    independent_stations=1,
                                                    livetime=livetime,
                                                    solid_angle=solid_angle)

            print(upper_limits_draw/(units.GeV * units.cm**-2 * units.s**-1 * units.sr**-1))
            exit()

        upper_limits.append(upper_limits_draw)

        if calculate_slopes:

            excluded_slope_draw = get_excluded_slope(energy_centres, effective_volumes_draw,
                                                     min_energy=20*units.PeV)

            excluded_slopes.append(excluded_slope_draw)

    upper_limits = np.array(upper_limits)
    upper_limits = np.transpose(upper_limits)
    excluded_slopes = np.array(excluded_slopes)

    quantiles_abs = { 0 : 0, 1 : 0.682689, 2 : 0.954499, 3 : 0.997300 }
    quantiles = {}
    for i_quantile in range(-3, 4):

        quantiles[i_quantile] = 0.5 + np.sign(i_quantile) * quantiles_abs[np.abs(i_quantile)] / 2

    flux_quantiles = {}
    slope_quantiles = {}

    for i_quantile, eval_quantile in quantiles.items():

        flux_quantiles[i_quantile] = np.quantile(upper_limits, eval_quantile, axis=1)
        if calculate_slopes:
            slope_quantiles[i_quantile] = np.quantile(excluded_slopes, eval_quantile, axis=0)

    return np.nan_to_num(flux_quantiles), np.nan_to_num(slope_quantiles)


"""
This function helps us produce the plot you can see on several proposals and
papers.
"""
fig, ax = limit.get_E2_limit_figure(diffuse = True,
                    show_ice_cube_EHE_limit=True,
                    show_ice_cube_HESE_fit=False,
                    show_ice_cube_HESE_data=True,
                    show_ice_cube_mu=True,
                    show_anita_I_III_limit=True,
                    show_auger_limit=True,
                    show_neutrino_best_fit=True,
                    show_neutrino_best_case=True,
                    show_neutrino_worst_case=True,
                    show_ara=True,
                    show_grand_10k=False,
                    show_grand_200k=False,
                    show_radar=False)

filenames = ['data/effective_volumes/Veff_dipole_array_Bastille_secondaries.json']
labels = { filenames[0] : 'RNO-G 35 stations, 5 years' }
colours = { filenames[0] : 'red' }

decade_bin = 'full' # 'half' or 'full'
mode = 'approximate' # 'proper' or 'approximate'

calculate_excluded = True
if calculate_excluded:
    Ntries = 10000
else:
    Ntries = 10000

for volumes_file in filenames[:1]:

    with open(volumes_file, 'r') as f:
        fin = json.load(f)

    Veffs_15 = np.array( fin['1.50sigma']["Veffs"] ) * units.m3
    Veffs_15_unc = np.array( fin['1.50sigma']["Veffs_uncertainty"] ) * units.m3
    Veffs_25 = np.array( fin['2.50sigma']["Veffs"] ) * units.m3
    Veffs_25_unc = np.array( fin['2.50sigma']["Veffs_uncertainty"] ) * units.m3
    Veffs_2 = np.array( fin['2.00sigma']["Veffs"] ) * units.m3
    Veffs_2_unc = np.array( fin['2.00sigma']["Veffs_uncertainty"] ) * units.m3
    energy_centres = np.array( fin["energies"] ) * units.eV

    n_decades = 5
    n_bins_per_decade = 6
    if decade_bin == 'half':
        decade_factor = n_decades / 2
    elif decade_bin == 'full':
        decade_factor = n_decades

    if mode == 'approximate':
        upper_limits_E2_15 = get_E2_upper_limits(energy_centres, Veffs_15,
                                                 independent_stations=decade_factor, livetime=2/3*5*units.year)
        unc_E2_15 = get_flux_uncertainty(Veffs_15, Veffs_15_unc, upper_limits_E2_15)
        unc_E2_15_MC, excluded_15 = get_flux_unc_MC(energy_centres, Veffs_15, Veffs_15_unc,
                                                    independent_stations=decade_factor, livetime=2/3*5*units.year,
                                                    calculate_slopes=calculate_excluded, Ntries=Ntries)
        upper_limits_E2_25 = get_E2_upper_limits(energy_centres, Veffs_25,
                                                 independent_stations=decade_factor, livetime=2/3*5*units.year)
        unc_E2_25 = get_flux_uncertainty(Veffs_25, Veffs_25_unc, upper_limits_E2_25)
        unc_E2_25_MC, excluded_25 = get_flux_unc_MC(energy_centres, Veffs_25, Veffs_25_unc,
                                                    independent_stations=decade_factor, livetime=2/3*5*units.year,
                                                    calculate_slopes=calculate_excluded, Ntries=Ntries)
        upper_limits_E2_2 = get_E2_upper_limits(energy_centres, Veffs_2,
                                                 independent_stations=decade_factor, livetime=2/3*5*units.year)
        unc_E2_2 = get_flux_uncertainty(Veffs_2, Veffs_2_unc, upper_limits_E2_2)
        unc_E2_2_MC, excluded_2 = get_flux_unc_MC(energy_centres, Veffs_2, Veffs_2_unc,
                                                  independent_stations=decade_factor, livetime=2/3*5*units.year,
                                                  calculate_slopes=calculate_excluded, Ntries=Ntries)
        energies_plot = energy_centres
    elif mode == 'proper':
        log_energy_centres = np.log10(energy_centres)
        Veffs[Veffs == 0] = 1e-20
        log_Veffs = np.log10(Veffs)
        log_energy_avgs = []
        log_Veffs_avg = []

        if decade_bin == 'half':
            n_group = int(n_bins_per_decade/2)
            final_n_bins = int(n_decades * 2)
        elif decade_bin == 'full':
            n_group = n_bins_per_decade
            final_n_bins = n_decades

        for i_group in range(final_n_bins):

            log_energy_avg = np.mean(log_energy_centres[i_group*n_group:(i_group+1)*n_group+1])
            log_energy_avgs.append(log_energy_avg)

            log_Veff_avg = np.mean(log_Veffs[i_group*n_group:(i_group+1)*n_group+1])
            log_Veffs_avg.append(log_Veff_avg)

        energy_avgs = 10**np.array(log_energy_avgs)
        Veffs_avg = 10**np.array(log_Veffs_avg)
        print(len(energy_avgs), len(Veffs_avg))
        upper_limits_E2 = get_E2_upper_limits(energy_avgs, Veffs_avg,
                                              independent_stations=1, livetime=2/3*5*units.year)
        energies_plot = energy_avgs

    units_flux_plot = units.GeV / units.cm2 / units.s / units.sr

    min_bin = 8

    excluded_slope = get_excluded_slope(energy_centres, Veffs_2)
    print('Excluded slope', excluded_slope)

    plt.plot([], [], label=labels[volumes_file], linestyle='')

    no_unc = False
    if no_unc:
        unc_E2_15_MC = np.zeros_like(unc_E2_15_MC)
        unc_E2_25_MC = np.zeros_like(unc_E2_25_MC)
        unc_E2_2_MC = np.zeros_like(unc_E2_2_MC)

    plt.fill_between(energies_plot[min_bin:]/units.GeV,
                     (unc_E2_15_MC[0][min_bin:])/units_flux_plot,
                     (unc_E2_25_MC[0][min_bin:])/units_flux_plot, alpha=0.5,
                     linestyle='-', color='red', linewidth=1,
                     label=r'[$1.5\sigma_{noise}$, $2.5\sigma_{noise}$] trigger')

    plt.fill_between(energies_plot[min_bin:]/units.GeV,
                     (unc_E2_25_MC[0][min_bin:])/units_flux_plot,
                     (unc_E2_25_MC[2][min_bin:])/units_flux_plot, alpha=0.5,
                     linestyle='-', color='orange', linewidth=1,
                     label=r'95% CL contour')

    plt.fill_between(energies_plot[min_bin:]/units.GeV,
                     (unc_E2_15_MC[-2][min_bin:])/units_flux_plot,
                     (unc_E2_15_MC[0][min_bin:])/units_flux_plot, alpha=0.5,
                     linestyle='-', color='orange', linewidth=1)

    plt.fill_between(energies_plot[min_bin:]/units.GeV,
                     (unc_E2_2_MC[-2][min_bin:])/units_flux_plot,
                     (unc_E2_2_MC[2][min_bin:])/units_flux_plot,
                     color='black', alpha=0.5,
                     label=r'$2\sigma_{noise}$ trigger')

    plt.plot(energies_plot[min_bin:]/units.GeV,
             (unc_E2_2_MC[0][min_bin:])/units_flux_plot,
             color='black')

    if calculate_excluded:

        print("Excluded slopes for 1.5 sigma")
        print(excluded_15)
        print("Excluded slopes for 2 sigma")
        print(excluded_2)
        print("Excluded slopes for 2.5 sigma")
        print(excluded_25)

        plt.fill_between(energies_plot[min_bin:]/units.GeV,
                         (ice_cube_nu_fit(energies_plot[min_bin:], excluded_15[-2]))*energies_plot[min_bin:]**2/units_flux_plot,
                         (ice_cube_nu_fit(energies_plot[min_bin:], excluded_25[2]))*energies_plot[min_bin:]**2/units_flux_plot,
                         color='mediumorchid', alpha=0.5,
                         label=r'IceCube-like flux')

        plt.plot(energies_plot[min_bin:]/units.GeV,
                         (ice_cube_nu_fit(energies_plot[min_bin:], excluded_2[0]))*energies_plot[min_bin:]**2/units_flux_plot,
                         color='mediumorchid', linestyle='--')


ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
handles = [handle for handle in handles if handle.get_label().find('RNO') != -1 or
           handle.get_label().find('noise') != -1 or handle.get_label().find('CL') != -1 or
           handle.get_label().find('IceCube-like') != -1]
plt.legend(handles=handles, loc=2)
plt.ylim((None, 3e-6))
plt.savefig('Sensitivities_RNO-35_1km_50cm_dipole.pdf', format='pdf')
plt.show()
