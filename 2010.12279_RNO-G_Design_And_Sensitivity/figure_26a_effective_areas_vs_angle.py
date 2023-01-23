#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'large'

import json
from glob import glob

import pandas as pd
from NuRadioMC.utilities import cross_sections
from NuRadioReco.utilities import units

files = ["data/effective_volumes/Veff_50deg.json",
         "data/effective_volumes/Veff_60deg.json",
         "data/effective_volumes/Veff_70deg.json",
         "data/effective_volumes/Veff_80deg.json"]


print(files)

veff_aeff_array = {}

### The ranges of angles can be found from the hdf5 input:


#import h5py
#angle_files = glob("33_RNO_angular/*part*")
#angle_files.sort()

#print("# file    \tthetamin\tthetamax")
#for f in angle_files:
#    inf = h5py.File(f)
#    print(f.split("_")[2], np.degrees(inf.attrs["thetamin"]), np.degrees(inf.attrs["thetamax"]))

## file     thetamin    thetamax
#angular/0.81 46.14882056822252 53.644924008225566
#angular/0.99 56.63298703076824 63.256316049597
#angular/1.17 66.91974284894391 73.02106216953294
#angular/1.35 77.07660112003705 82.89730354962327

angle_range = {80: [77,83],
               70: [67,73],
               60: [57,63],
               50: [46,54]}

colors = {50: "green", 60: "blue", 70: "purple", 80: "red"}
linestyles = {50: "--", 60: "-.", 70: "-", 80: ":"}


for plot_sigma_range in [True, False]:
    plt.figure()
    for filename in files:
        print(filename)
        with open(filename) as file:
            data = json.load(file)

            V = np.array(data['2.00sigma']["Veffs"])
            A = V/cross_sections.get_interaction_length(np.array(data["energies"]))

            Vmin = np.array(data['2.50sigma']["Veffs"])
            Amin = Vmin/cross_sections.get_interaction_length(np.array(data["energies"]))
        
            Vmax = np.array(data['1.50sigma']["Veffs"])
            Amax = Vmax/cross_sections.get_interaction_length(np.array(data["energies"]))
        
            data["veffs"] = V
            data["aeffs"] = A
            degrees = int(filename.replace("deg.json", "").split("_")[-1])
            label = r'$\theta_z$ = [' + str(angle_range[degrees][0]) + r"$^{\circ}$, " + str(angle_range[degrees][1]) + r"$^{\circ}$]"
            plt.plot(np.array(data["energies"]), A/units.km**2, linewidth=2, label=label, color=colors[degrees], linestyle=linestyles[degrees]) 
            if plot_sigma_range:
                plt.fill_between(np.array(data["energies"]), Amin/units.km**2, Amax/units.km**2, alpha=0.2, color=colors[degrees])
    plt.loglog()
    plt.xlim(1e16, 1e20)
    plt.ylim(1e-5, 1e0)
    plt.legend()
    plt.xlabel("Neutrino Energy [eV]")
    plt.ylabel(r"Effective Area [km$^2$]")

    plt.tight_layout()
    if plot_sigma_range:
        plt.savefig("plots/effective_areas_RNO_G_1.5_2.5_sigmaBand.pdf")
    else:
        plt.savefig("plots/effective_areas_RNO_G.pdf")



