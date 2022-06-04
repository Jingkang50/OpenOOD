# srun -p dsta --mpi=pmi2 --cpus-per-task=1
# --kill-on-bad-exit=1 --job-name=tsne -w SG-IDC1-10-51-2-73
# python compute_tsne.py

import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

l2_normalize = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)


def tsne_compute(x, n_components=50):
    start_time = time.time()
    if n_components < x.shape[1]:
        pca = PCA(n_components=50)
        x = pca.fit_transform(x)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000)
    tsne_pos = tsne.fit_transform(x)

    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('TSNE Computation Duration: {:0>2}:{:0>2}:{:05.2f}'.format(
        int(hours), int(minutes), seconds),
          flush=True)

    return tsne_pos


dataset_list = [
    'mnist', 'usps', 'svhn', 'notmnist', 'fashionmnist', 'texture', 'cifar10',
    'tin'
]
dirname = '/mnt/lustre/jkyang/FSOOD22/report/test/test_tsne'
sample_rate = 0.1

highfeat_list, featstat_list, idx_list = [], [], []
for idx, dataset in enumerate(dataset_list):
    file_name = os.path.join(dirname, f'{dataset}.npz')
    highfeat_sublist = np.load(file_name)['highfeat_list']
    featstat_sublist = np.load(file_name)['featstat_list']
    # label_list = np.load(file_name)['label_list']
    # selection:
    num_samples = len(highfeat_sublist)
    index_list = np.arange(num_samples)
    index_select = np.random.choice(index_list,
                                    int(sample_rate * num_samples),
                                    replace=False)
    highfeat_list.extend(highfeat_sublist[index_select])
    featstat_list.extend(featstat_sublist[index_select])
    idx_list.extend(idx * np.ones(len(index_select)))

highfeat_list, featstat_list, index_list = np.array(highfeat_list), np.array(
    featstat_list), np.array(idx_list)
tsne_pos_highfeat = tsne_compute(highfeat_list)
tsne_pos_lowfeat = tsne_compute(featstat_list)
np.save(os.path.join(dirname, 'tsne_pos_highfeat'), tsne_pos_highfeat)
np.save(os.path.join(dirname, 'tsne_pos_lowfeat'), tsne_pos_lowfeat)
np.save(os.path.join(dirname, 'idx'), idx_list)
