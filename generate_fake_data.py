
# coding: utf-8

# In[1]:

import pandas as pd
import nibabel as nb
import numpy as np
import os
from nilearn.image import resample_img, smooth_img
from scipy import stats
import joblib
import sys

data_location = sys.argv[1]
print(data_location)
n_fake_maps = int(sys.argv[2])


# In[2]:

df = pd.read_csv(data_location+"/smoothness_and_volume.csv", index_col=0)


# In[3]:

import numpy as np

def hist_match(source, t_quantiles, t_values):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


# In[4]:

def save_fake_map(orig_inside_mask, map_nii, row, mask_nii, t_quantiles, t_values, i, j):
    np.random.seed(j)
    new_data = np.random.choice(orig_inside_mask, size=map_nii.shape, replace=True)

    shuffled_nii = nb.Nifti1Image(new_data, map_nii.affine, map_nii.header)
    smoothed_nii = smooth_img(shuffled_nii, np.array([row.FWHMx_mm, row.FWHMy_mm, row.FWHMz_mm]))
    new_data = smoothed_nii.get_data()
    new_data[mask_nii.get_data() != 1] = np.nan
    new_inside_mask = stats.zscore(new_data[mask_nii.get_data() == 1])

    new_data[mask_nii.get_data() == 1] = new_inside_mask #hist_match(new_inside_mask, t_quantiles, t_values)

    masked_nii = nb.Nifti1Image(new_data, map_nii.affine, map_nii.header)

    masked_nii.to_filename(data_location +"/images/fake_maps/%04d/%04d.nii.gz"%(i, j))


# In[5]:

get_ipython().system('mkdir -p '+ data_location + '/images/fake_maps')

for i,row in df.iterrows():
    print("generating fake data for map images/resampled/%04d.nii.gz"%i)
    map_nii = nb.load(data_location+"/images/resampled/%04d.nii.gz"%i)
    mask_nii = nb.load(data_location+"/images/resampled_masks/%04d.nii.gz"%i)

    orig_inside_mask = map_nii.get_data()[mask_nii.get_data() == 1]

    folder_name = "%04d"%i
    if os.path.exists(data_location+"/images/fake_maps/%04d/"%(i)):
        continue
    get_ipython().system('mkdir -p '+data_location+'/images/fake_maps/{"%04d"%i}')

    template = orig_inside_mask.ravel()

    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    from joblib import Parallel, delayed
    Parallel(n_jobs=16)(delayed(save_fake_map)(orig_inside_mask, map_nii, row, mask_nii, t_quantiles, t_values, i, j) for j in range(n_fake_maps))
