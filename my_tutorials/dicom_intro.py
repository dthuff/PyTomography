import os
import numpy as np
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.utils import print_collimator_parameters
import matplotlib.pyplot as plt
import pydicom
import shutil

# CHANGE THIS TO WHERE YOU DOWNLOADED THE TUTORIAL DATA
root_tutorial_data_path = '/home/daniel/datasets/pytomo'

# initialize the `path`` variable below to specify the location of the required data
path = os.path.join(root_tutorial_data_path, 'dicom_intro_tutorial-20241128T012321Z-001', 'dicom_intro_tutorial')
path_CT = os.path.join(path, 'CT_files')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
file_NM = os.path.join(path, 'projection_data.dcm')

object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)

photopeak = dicom.get_projections(file_NM, index_peak=0)
print(photopeak.shape)

plt.figure(figsize=(7,6))
plt.imshow(photopeak[0].cpu().T, cmap='nipy_spectral', origin='lower')
plt.axis('off')
plt.colorbar(label='Counts')

scatter = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)

att_transform = SPECTAttenuationTransform(filepath=files_CT)

att_transform.configure(object_meta, proj_meta)
attenuation_map = att_transform.attenuation_map

sample_slice = attenuation_map.cpu()[:,70].T

plt.subplots(1,2,figsize=(7,3))
plt.subplot(121)
plt.title('Projection Data')
plt.imshow(photopeak[0].cpu().T, cmap='nipy_spectral', origin='lower')
plt.axis('off')
plt.colorbar(label='Counts')
plt.subplot(122)
plt.title('From CT Slices')
plt.imshow(sample_slice, cmap='Greys_r', origin='lower', interpolation='Gaussian')
plt.axis('off')
plt.colorbar()
plt.show()

a=4