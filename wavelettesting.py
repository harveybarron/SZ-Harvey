from params import khatrix
import pymaster as nmt
import pickle
import matplotlib.pyplot as plt
import numpy as np

arcmin2kpc = 27.052
khatrix_log = tuple([np.log10(x) for x in khatrix])
khatrix_log_diff =  [(t-s) for s, t in zip(khatrix_log, khatrix_log[1:])]
ells_uncoupled_khatri = tuple([x * arcmin2kpc * 60 * 180 for x in khatrix])

ells_uncoupled_khatri_diff = [10 ** ((np.log10(s) + np.log10(t)) / 2) for s, t in
                              zip(ells_uncoupled_khatri, ells_uncoupled_khatri[1:])]
khatri_diff = [10 ** ((np.log10(s) + np.log10(t)) / 2) for s, t in
                              zip(khatrix, khatrix[1:])]
ells_uncoupled_khatri_diff1 = tuple([x * arcmin2kpc * 60 * 180 for x in khatri_diff])
print(ells_uncoupled_khatri_diff)
print(ells_uncoupled_khatri_diff1)

l0_bins_klow = []
lf_bins_klow = []
for i in range(5):
    if i == 0:
        l0_bins_klow.append(2 * ells_uncoupled_khatri[i] - ells_uncoupled_khatri_diff[i])
    else:
        l0_bins_klow.append(lf_bins_klow[i - 1])
    lf_bins_klow.append(ells_uncoupled_khatri_diff[i])

l0_bins_klow.append(lf_bins_klow[4])
lf_bins_klow.append(2* ells_uncoupled_khatri[5] - ells_uncoupled_khatri_diff[4])

# l0_bins_klog = []
# lf_bins_klog = []
# for i in range(6):
#     if i == 0:
#         l0_bins_klog.append(khatrix_log[i] - khatrix_log_diff[i] / 2)
#     else:
#
print("l0bins k")
print(l0_bins_klow)
print("lfbins k")
print(lf_bins_klow)
bklow = nmt.NmtBinFlat(l0_bins_klow, lf_bins_klow)
ells_uncoupled_bklow = bklow.get_effective_ells()
print(ells_uncoupled_bklow)
print(ells_uncoupled_khatri)