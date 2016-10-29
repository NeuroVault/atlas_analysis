from pyneurovault import api
import pandas as pd
from nilearn.image import resample_img, smooth_img
import nibabel as nb
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy import stats
from scipy.ndimage.measurements import labeled_comprehension
import sys

atlas_id = sys.argv[2]
data_location = sys.argv[1]

atlases = api.get_images(pks=[atlas_id])
api.download_images(data_location + "/atlases", atlases, resample=False)

standard = "/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"

get_ipython().system('mkdir -p '+data_location+'/atlases/resampled')

fname = atlas_id+".nii.gz"
print "resampling " + fname
nii = nb.load(data_location+"atlases/original/" + fname)

target_nii = nb.load(standard)
resampled_nii = resample_img(nii,target_affine=target_nii.get_affine(),
                             target_shape=target_nii.shape, interpolation='nearest')
resampled_nii.to_filename(data_location+"atlases/resampled/" + fname)

def score_map(map_filename, atlas_data, labels):
    map_data = nb.load(map_filename).get_data()
    parcel_means = labeled_comprehension(map_data,
                                             atlas_data,
                                             list(labels),
                                             np.mean,
                                             float,
                                             np.nan)
    parcel_variance = labeled_comprehension(map_data,
                                             atlas_data,
                                             list(labels),
                                             np.var,
                                             float,
                                             np.nan)
    within_variance = parcel_variance.mean()
    between_variance = parcel_means.var()
    return within_variance, between_variance

comparisons_df = pd.DataFrame(columns=["within_parcel_variance", "between_parcel_variance", "atlas_id", "image_id", "fake_map_id"])

df = pd.read_csv(data_location + "/smoothness_and_volume.csv", index_col=0)

n_fake_maps = int(sys.argv[3])

for i, atlas_row in atlases.iterrows():
    atlas_id = atlas_row["image_id"]
    print "analyzing " + atlas_row["name"]
    atlas_nii = nb.load(data_location + "/atlases/resampled/%04d.nii.gz"%atlas_id)

    for image_id, row in df.iterrows():
        fname = "%04d.nii.gz"%image_id
        print "analyzing " + fname
        mask_nii = nb.load(data_location + "/images/resampled_masks/%04d.nii.gz"%image_id)
        atlas_data = atlas_nii.get_data().copy()
        atlas_data[mask_nii.get_data() != True] = 0
        labels = set(np.unique(atlas_data)) - set([0])

        if not ((comparisons_df['atlas_id'] == atlas_id) & (comparisons_df['image_id'] == image_id)).any():
            real_within_variance, real_between_variance = score_map(data_location + "/images/resampled/%04d.nii.gz"%image_id, atlas_data, labels)
            comparisons_df = comparisons_df.append({"within_parcel_variance": real_within_variance,
                                   "between_parcel_variance": real_between_variance,
                                   "atlas_id": atlas_id,
                                   "image_id": image_id,
                                   "fake_map_id": None}, ignore_index=True)

        from joblib import Parallel, delayed
        out = Parallel(n_jobs=2)(delayed(score_map)(data_location + "/images/fake_maps/%04d/%04d.nii.gz"%(image_id, fake_map_id),
                                                                                   atlas_data,
                                                                                   labels) for fake_map_id in range(n_fake_maps))
        comparisons_df = comparisons_df.append(pd.DataFrame({"within_parcel_variance": list(np.array(out)[:, 0]),
                                       "between_parcel_variance": list(np.array(out)[:, 1]),
                                       "atlas_id": [atlas_id]*n_fake_maps,
                                       "image_id": [image_id]*n_fake_maps,
                                       'fake_map_id':range(n_fake_maps)}), ignore_index=True)

comparisons_df.to_csv(data_location + "/%scomparison.csv"%sys.argv[2])
