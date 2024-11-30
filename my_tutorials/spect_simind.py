# Taken from https://pytomography.readthedocs.io/en/latest/notebooks/t_siminddata.html#3-Likelihood-/-Reconstruction

import inspect
import os

import matplotlib.pyplot as plt
import torch

from pytomography.algorithms import OSEM
from pytomography.io.SPECT import simind
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform

# CHANGE THIS TO WHERE YOU DOWNLOADED THE TUTORIAL DATA
PATH = '/home/daniel/datasets/pytomo/mc_tutorials-20241128T012320Z-001/'
data_path = os.path.join(PATH, 'mc_tutorials', 'lu177_SYME_jaszak_lowres')

photopeak_path = os.path.join(data_path, 'tot_w4.h00')
lower_path = os.path.join(data_path, 'tot_w5.h00')
upper_path = os.path.join(data_path, 'tot_w6.h00')

activity = 1000  # MBq
time_per_proj = 15  # s

object_meta, proj_meta = simind.get_metadata(photopeak_path)
photopeak = simind.get_projections(photopeak_path)
photopeak_realization = torch.poisson(photopeak * activity * time_per_proj)

"""
# Plot Expected projection data for some angles
fig, ax = plt.subplots(1, 4, figsize=(8, 2.4), gridspec_kw={'wspace': 0.0})
for i in range(4):
    ax[i].imshow(photopeak[10 * i].cpu().T, cmap='magma')
    ax[i].axis('off')
    ax[i].text(0.03, 0.97, f'Angle: {10 * proj_meta.angles[i]}dg', ha='left', va='top', color='white',
               transform=ax[i].transAxes, fontsize=12)
plt.show()
"""

"""
# Plot Real projection data for some angles - this is noisier
fig, ax = plt.subplots(1, 4, figsize=(8, 2.4), gridspec_kw={'wspace': 0.0})
for i in range(4):
    ax[i].imshow(photopeak_realization[10 * i].cpu().T, cmap='magma')
    ax[i].axis('off')
    ax[i].text(0.03, 0.97, f'Angle {10 * i}', ha='left', va='top', color='white', transform=ax[i].transAxes,
               fontsize=12)
plt.show()
"""

lower = simind.get_projections(lower_path)
upper = simind.get_projections(upper_path)
lower_realization = torch.poisson(lower * activity * time_per_proj)
upper_realization = torch.poisson(upper * activity * time_per_proj)

ww_peak, ww_lower, ww_upper = [simind.get_energy_window_width(path) for path in
                               [photopeak_path, lower_path, upper_path]]
scatter_estimate_TEW = simind.compute_EW_scatter(lower_realization, upper_realization, ww_lower, ww_upper, ww_peak)

# Plot scatter estimates from TEW method
"""
fig, ax = plt.subplots(1, 4, figsize=(8, 2.4), gridspec_kw={'wspace': 0.0})
for i in range(4):
    ax[i].imshow(scatter_estimate_TEW[10 * i].cpu().T, cmap='magma')
    ax[i].axis('off')
    ax[i].text(0.03, 0.97, f'Angle {10 * i}', ha='left', va='top', color='white', transform=ax[i].transAxes,
               fontsize=12)
"""
path_amap = os.path.join(data_path, 'amap.hct')
amap = simind.get_attenuation_map(path_amap)

"""
plt.figure(figsize=(3, 3))
plt.imshow(amap[:, :, 64].cpu().T, cmap='Greys_r')
plt.axis('off')
plt.colorbar(label='cm^-1')
plt.show()
"""

# Get PSF
att_transform = SPECTAttenuationTransform(amap)
psf_meta = simind.get_psfmeta_from_header(photopeak_path)
# Below is optional to print details about psf_meta
sigma_fit_func_code = inspect.getsource(psf_meta.sigma_fit)
print(psf_meta)
print(sigma_fit_func_code)

psf_transform = SPECTPSFTransform(psf_meta)

# Create sys matrix
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[att_transform, psf_transform],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta
)

likelihood = PoissonLogLikelihood(
    system_matrix=system_matrix,
    projections=photopeak_realization,
    additive_term=scatter_estimate_TEW
)

recon_algorithm = OSEM(likelihood)

# The reconstructed image
n = 5
fig, ax = plt.subplots(n, n, figsize=(8, 8), gridspec_kw={'wspace': 0.0})

for i in range(n):
    for j in range(n):
        reconstructed_image = recon_algorithm(
            n_iters=4 * (i + 1),
            n_subsets=2 * (j + 1),
        )
        ax[i, j].imshow(reconstructed_image[:, :, 64].cpu().T, cmap='magma', origin='lower', interpolation='Gaussian')
        ax[i, j].set_xlim(40, 85)
        ax[i, j].set_ylim(40, 85)
        ax[i, j].set_title(f'{str(4 * (i + 1))}it{str(4 * (j + 1))}ss')
        ax[i, j].axis('off')

plt.show()
