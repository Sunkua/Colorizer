import glob
import numpy as np
from scipy.ndimage.filters import gaussian_filter

a_hists = glob.glob('/network-ceph/pgrundmann/maschinelles_sehen/histogram_mean/np_means/*hist_a.np.npy')
b_hists = glob.glob('/network-ceph/pgrundmann/maschinelles_sehen/histogram_mean/np_means/*hist_b.np.npy')
test = "a"

np_a_hists = []
for hist in a_hists:
    np_a_hists.append(np.load(hist))

np_b_hists = []
for hist in b_hists:
    np_b_hists.append(np.load(hist))

a_histogram = sum(np_a_hists)
b_histogram = sum(np_b_hists)
a_histogram /= len(np_a_hists)
b_histogram /= len(np_b_hists)


# merged-shape: a,b
merged_histogram = np.dstack(np.meshgrid(a_histogram, b_histogram)).reshape(-1, 2)
smoothed = gaussian_filter(merged_histogram, sigma=5)
smoothed_mixed = (0.5 * smoothed) + (0.5 / (a_histogram.shape[0] * b_histogram.shape[0]))
reciprocal = np.power(smoothed_mixed, -1)
reciprocal /= reciprocal.sum()
test = 2

