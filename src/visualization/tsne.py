import multiprocessing as mp
import os
import nilearn.image
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from models.dataloaders import *
import pandas as pd
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn import manifold


def print_mem_of_array(x):
    bytes = x.size * x.itemsize
    print(f"{np.round(bytes/1e+9,2)} Gb")


def plot_tsne(x,label):


    plt.figure(figsize=(8, 8))

    for color, i, target_name in zip(['r', 'b'], [0, 1], ["control", "dependent"]):
        plt.scatter(x[label == i, 0], x[label == i, 1],
                    color=color, lw=2, label=target_name)

    #plt.set_title("Perplexity=%d" % perplexity)
    #ax.axis('tight')
    plt.show()



def find_balanced_subsets(N,path):

    df = pd.read_csv(path)
    labels = []
    dfg = df.groupby("dependent")
    data = []
    for name, subdata in dfg:
        print(f"Group: {name}")
        data.extend(subdata["filename"][:N].values)
        labels.extend(subdata["dependent"][:N].values)

    # shuffle
    data = np.array(data).reshape(-1)
    labels=np.array(labels).reshape(-1)
    ind = np.random.choice(len(labels),len(labels),replace=False)

    labels = labels[ind]
    data= data[ind]

    assert data.shape[0] == labels.shape[0]

    return data, labels



def ipca_transform(ipca,batch):
    #x, label = batch
    x = batch['image'][tio.DATA]
    label = batch['label']
    batch_size = x.shape[0]
    partial_x = x.squeeze().reshape(batch_size, -1)
    #label.append(label.numpy())
    partial_x_transform = ipca.transform(partial_x)
    #x_transform.append(partial_x_transform)
    # x_transform = np.vstack((x_transform, partial_x_transform))
    return (partial_x_transform, label)


def multiprocess_transform(ipca, dataloader, batch_size, ncpus=None):


    if ncpus is None:
        ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))

    x_transform = []
    label = []

    pool = mp.Pool(processes=ncpus)

    print(f"Multiprocessing using {ncpus} cpus")
    results = [pool.apply_async(ipca_transform, args=(ipca,batch,)) for batch in dataloader]
    #x_transform = [p.get() for p in results]

    for p in results:
        res = p.get()
        x_transform.append(res[0])
        label.append(res[1])

    x_transform = np.concatenate(x_transform)
    label = np.concatenate(label)

    return x_transform, label


def iterative_pca(batch_size,dataloader,n_components,out_name):

    if n_components > batch_size:
        print(f"n_components can't be smaller than batch_size, resizing to n_components = batch_size.")
        n_components = batch_size

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # compute necessary paramters iteratively
    print("Fitting IPCA...")
    for i, batch in enumerate(dataloader):
        x = batch['image'][tio.DATA]
        y = batch['label']
        data = x.squeeze().reshape(x.shape[0], -1)
        print(f"iter: {i}, data shape: {data.shape}")
        ipca.partial_fit(data)

    print("Transforming input using estimates from fit...")
    x_transform, label = multiprocess_transform(ipca, dataloader, batch_size, ncpus=None)

    print("Saving result under data directory.")
    np.save(f'data/x_{out_name}.npy', x_transform, allow_pickle=True)
    np.save(f'data/label_{out_name}.npy', label, allow_pickle=True)

    return x_transform, label



def tsne(path,N,batch_size,num_workers,out_name, clobber=True):

    if not os.path.exists(f"data/x_{out_name}.npy") or clobber:

        image_paths, labels = find_balanced_subsets(N, path)

        indices = np.arange(len(labels))

        dm = MRIDataModuleIO(path, labels, 'nifti', batch_size, None, '', image_paths, num_workers)
        dm.prepare_data()
        dm.setup(stage='fit')

        x_transform, label = iterative_pca(batch_size, dm.train_dataloader(), 25, out_name)

    else:
        print("Files located, loading in progres...")
        x_transform = np.load(f'data/x_{out_name}.npy', allow_pickle=True)
        label =  np.load(f'data/label_{out_name}.npy', allow_pickle=True)


    print(f"Shape of input: {x_transform.shape}")
    print(f"Shape of labels: {label.shape}")

    print(f"Size of input data on disk")
    print_mem_of_array(x_transform)

    #  tsne
    perplexities = [20]
    for i, perplexity in enumerate(perplexities):
        #ax = subplots[0][i + 1]

        #t0 = time()
        tsne = manifold.TSNE(n_components=2, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(x_transform)
        #t1 = time()

        print("Completed and plotting results...")
        plt.clf()
        plot_tsne(Y, label)
        plt.savefig(f"tsne_mri_p{perplexity}.png")

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/Users/sean/Projects/MRI_Deep_Learning/Kamran_Montreal_Data_Share/data_split_test_c.csv')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=12)
    # parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--name', type=str, default='tsne')
    parser.add_argument('--out_name', type=str, default='test')
    parser.add_argument('--cropped', type=bool, default=False)
    parser.add_argument('--clobber', type=bool, default=False)
    parser.add_argument('--step_size', type=float, default=0.001)
    args = parser.parse_args()


    tsne(args.data_dir, args.max_samples, args.batch_size, args.num_workers, args.out_name,clobber=True)
