import os
import pickle
from PIL import Image
import h5py
import json
import tarfile

root = 'miniimagenet'
# Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
gz_filename = 'mini-imagenet.tar.gz'
gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
pkl_filename = 'mini-imagenet-cache-{0}.pkl'

filename = '{0}_data.hdf5'
filename_labels = '{0}_labels.json'


for split in ['train', 'val', 'test']:
    filename_t = os.path.join(root, filename.format(split))
    if os.path.isfile(filename_t):
        continue

    pkl_filename_t = os.path.join(root, pkl_filename.format(split))
    if not os.path.isfile(pkl_filename_t):
        raise IOError()
    with open(pkl_filename_t, 'rb') as f:
        data = pickle.load(f)
        images, classes = data['image_data'], data['class_dict']

    with h5py.File(filename_t, 'w') as f:
        group = f.create_group('datasets')
        for name, indices in classes.items():
            group.create_dataset(name, data=images[indices])

    labels_filename = os.path.join(root, filename_labels.format(split))
    with open(labels_filename, 'w') as f:
        labels = sorted(list(classes.keys()))
        json.dump(labels, f)

    if os.path.isfile(pkl_filename_t):
        os.remove(pkl_filename_t)