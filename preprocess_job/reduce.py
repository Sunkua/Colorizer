import glob
import numpy as np
import h5py
import hdf5plugin
from tqdm import tqdm
hdf5_files = glob.glob('/network-ceph/pgrundmann/youtube_precalculated/dataset_*.hdf5')


with h5py.File('/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5', 'w') as f:
    files = []
    for dataset in tqdm(hdf5_files):
        with h5py.File(dataset, 'r') as f_data:
            for name in f_data:
                if not name in f:
                    f_data.copy(name, f) 
                '''
                grp = f.create_group(name)
                ds_l_source = f_data[name+"/images_l"]
                ds_images_hists_source = f_data[name+'/images_hists']
                ds_resnet_source = f_data[name+"/images_resnet"]
                #l_ds = grp.create_dataset('images_l', ds_l_source.shape, dtype=np.float32, data=ds_l_source)
                #histogram_ds = grp.create_dataset('images_hists', ds_images_hists_source.shape, dtype=np.float32, data=ds_images_hists_source)
                #resnet_ds = grp.create_dataset('images_resnet', ds_resnet_source.shape, dtype=np.float32, data=ds_resnet_source)
                l_ds = grp.create_dataset('images_l', ds_l_source.shape, dtype=np.float32, data=ds_l_source,**hdf5plugin.Blosc(cname='blosclz', clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
                histogram_ds = grp.create_dataset('images_hists', ds_images_hists_source.shape, dtype=np.float32, data=ds_images_hists_source, **hdf5plugin.Blosc(cname='blosclz', clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
                resnet_ds = grp.create_dataset('images_resnet', ds_resnet_source.shape, dtype=np.float32, data=ds_resnet_source, **hdf5plugin.Blosc(cname='blosclz', clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
                '''
