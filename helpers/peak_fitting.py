import numpy as np
from helpers.physics_functions import get_bins_for_scan


known_resonances_OS = {
    "eta": 0.547862,
    "rho": 0.77545,
    "phi": 1.019461,
    "J_psi": 3.096916,
    "psi_prime": 3.686097,
    "upsilon_1S": 9.46040,
    "upsilon_2S": 10.0234,
    "upsilon_3S": 10.3551,
    }


known_resonances_SS = {
    }



def check_for_known_resonances(resonance_dict, left, right):

    return_dict = {}
    for resonance, mass in resonance_dict.items():
        if left < mass < right:
            return_dict[resonance] = mass

    return return_dict


def known_resonances_in_window(BIN_INFO, mass_scaler, multiplier = 1.1, include_SR = False, train_samesign = False, real_space = True):

    """
    Make sure the band boundaries are in the same space (real, scaled) as the flag says they are!
    Technically we only need mass_scaler if working in scaled space
    """

    # Bin info
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = BIN_INFO
    SR_left, SR_right = plot_bins_SR[0], plot_bins_SR[-1]
    SB_left, SB_right = plot_bins_left[0], plot_bins_right[-1]

    if train_samesign:
        known_resonances = known_resonances_SS
    else:
        known_resonances = known_resonances_OS

    if real_space:
        SB_left = SB_left / multiplier
        SB_right = SB_right * multiplier
        SR_left = SR_left / multiplier
        SR_right = SR_right * multiplier
  
    else:
        SB_left = SB_left * multiplier
        SB_right = SB_right * multiplier
        SR_left = SR_left * multiplier
        SR_right = SR_right * multiplier

        for resonance, mass in known_resonances.items():
            known_resonances[resonance] = mass_scaler.transform(np.array([[mass]]))[0][0]

    in_full_range = check_for_known_resonances(known_resonances, SB_left, SB_right)
    in_SR = check_for_known_resonances(known_resonances, SR_left, SR_right)

    if include_SR:
        return in_full_range
    else:

        # Get the dictionary of things in the full range but not in the SR
        in_full_not_SR = {}
        for key in in_full_range.keys():
            if key not in in_SR.keys():
                in_full_not_SR[key] = in_full_range[key]

        return in_full_not_SR


