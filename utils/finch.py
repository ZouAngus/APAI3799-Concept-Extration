import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import ot
import miniball
import warnings
# from utils.ptp_utils import kld_distance, emd_distance_2d, get_connect, high_dim_connect

try:
    from  import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

def clust_rank(
        mat,
        use_ann_above_samples,
        initial_rank=None,
        distance='cosine',
        verbose=False):

    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = np.empty(shape=(1, 1))
    elif s <= use_ann_above_samples:
        if distance != 'kld' and distance != 'emd':
            orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        elif distance == 'kld':
            orig_dist = kld_distance(mat, mat)
        elif distance == 'emd':
            orig_dist = emd_distance_2d(mat, mat)
            
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(use_ann_above_samples))
        if verbose:
            print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None, connect=None):
    if min_sim is not None:
        a[np.where(orig_dist > min_sim)] = 0
    
    if connect is not None:
        a[np.where(connect == 0)] = 0
    
    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance, use_ann_above_samples, verbose):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, use_ann_above_samples, initial_rank=None, distance=distance, verbose=verbose)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_

def kld_distance(mat1, mat2):
    def compute_kld(mat1, mat2):
        plogp = np.sum(mat1 * np.log(mat1), axis=-1, keepdims=True) # P, 1
        plogq = mat1 @ np.log(mat2).T # P, Q
        
        kld = plogp - plogq # P,Q
        
        return kld
    
    kld_pq = compute_kld(mat1, mat2)
    kld_qp = compute_kld(mat2, mat1).T
    
    kld = (kld_pq + kld_qp) / 2
    
    return kld


#### Utilities from ptp_utils.py
def emd_distance_2d(mat1, mat2): # input: B, h*w
    b, hw = mat1.shape
    h = int(hw**0.5)
    
    mat1 = mat1 / mat1.sum(-1, keepdims=True)
    mat2 = mat2 / mat2.sum(-1, keepdims=True)
    
    X, Y = np.meshgrid(np.arange(h), np.arange(h))
    grid = np.stack((Y.flatten(), X.flatten()), axis=-1)

    cost = ot.dist(grid, metric='euclidean', p=2)
    
    dist = np.zeros((b, b))
    for i in range(b-1):
        for j in range(i+1, b):
            dist[i, j] = ot.emd2(mat1[i], mat2[j], cost, 1)
            dist[j, i] = dist[i, j]
            
    return dist
def get_connect(mask_mat):
    b, h, w = mask_mat.shape
    
    connect = np.zeros((b, b))
    for i in range(b-1):
        for j in range(i+1, b):
            num1 = get_num_mask(mask_mat[i])
            num2 = get_num_mask(mask_mat[j])
            num_sum = num1 + num2
            
            num = get_num_mask(mask_mat[i] + mask_mat[j])
            if num < num_sum: # only one mask
                connect[i,j] = 1
                connect[j,i] = 1
            else: # multi masks
                connect[i,j] = 0
                connect[j,i] = 0
    
    return connect


def high_dim_connect(mask_mat, orig_map):
    b = mask_mat.shape[0]
    rng = np.random.default_rng(42)  
           
    pca = PCA(n_components=100)
    mat_pca = pca.fit_transform(orig_map)
            
    mask_mat = mask_mat.reshape(b, -1)
    
    connect = np.ones((b, b))
    for i in range(b-1):
        for j in range(i+1, b):
            set1 = mat_pca[mask_mat[i] == 1]
            set2 = mat_pca[mask_mat[j] == 1]

            c1, r1 = miniball.get_bounding_ball(set1*100, rng = rng)
            c2, r2 = miniball.get_bounding_ball(set2*100, rng = rng)
            
            if  sum((c1 - c2)**2) > (r1 + r2)**2:
                connect[i,j] = 0
                connect[j,i] = 0
                
    assert False
    return connect

#### main function for FINCH
def FINCH(
        data,
        initial_rank=None,
        req_clust=None,
        distance='cosine',
        ensure_early_exit=True,
        verbose=True,
        use_ann_above_samples=70000,
        min_sim=None,
        mask_candidate=None,
        orig_sim_map=None,
        update_min_sim = False
        ):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :param use_ann_above_samples: Above this data size (number of samples) approximate nearest neighbors will be used to speed up neighbor
        discovery. For large scale data where exact distances are not feasible to compute, set this. [default = 70000]
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    # Cast input data to float32
    data = data.astype(np.float32)

    adj, orig_dist = clust_rank(data,
                                use_ann_above_samples,
                                initial_rank,
                                distance,
                                verbose)
    initial_rank = None
    
    if mask_candidate is not None:
        mask_candidate_in = mask_candidate
        connect = get_connect(mask_candidate)
        # if orig_sim_map is not None:
        #     connect_high_dim = high_dim_connect(mask_candidate, orig_sim_map)
    else:
        connect = None

    group, num_clust = get_clust(adj, orig_dist, min_sim, connect)
    c, mat = get_merge([], group, data)

    
    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    # if ensure_early_exit:
    #     if orig_dist.shape[-1] > 2:
    #         min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    
    if mask_candidate is not None:
        mask_list = []
        for i in range(c_.max()+1):
            mask_i = mask_candidate_in[c_==i].sum(axis=0)
            mask_list.append(mask_i)
        mask_candidate = np.stack(mask_list, axis=0)
            
    k = 1
    num_clust = [num_clust]
    
    min_sim_list = [min_sim]
    while exit_clust > 1:
        mat_orig = mat
        adj, orig_dist = clust_rank(mat, use_ann_above_samples, initial_rank, distance, verbose)
            
        if mask_candidate is not None:
            connect = get_connect(mask_candidate)
            
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim, connect)
        c_, mat = get_merge(c_, u, data)
        
        min_sim_list.append(np.max(orig_dist * adj.toarray()))
            
        if mask_candidate is not None:
            mask_list = []
            for i in range(c_.max()+1):
                mask_i = mask_candidate_in[c_==i].sum(axis=0)
                mask_list.append(mask_i)
            mask_candidate = np.stack(mask_list, axis=0)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            min_sim_list = min_sim_list[:-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
            
        k += 1
        
        # if mat_orig.shape[-1] > 2:
        #     s = mat_orig.shape[0]
        
        #     # distance == 'emd'
        #     dist_emd = emd_distance_2d(mat_orig, mat_orig)
        #     np.fill_diagonal(dist_emd, 1e12)
        #     min_sim = np.max(dist_emd * adj.toarray())
            
        
    if req_clust is not None:
        if req_clust not in num_clust:
            if req_clust > num_clust[0]:
                print(f'requested number of clusters are larger than FINCH first partition with {num_clust[0]} clusters . Returning {num_clust[0]} clusters')
                req_c = c[:, 0]
            else:
                ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
                req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, use_ann_above_samples, verbose)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c, min_sim_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Specify the path to your data csv file.')
    parser.add_argument('--output-path', default=None, help='Specify the folder to write back the results.')
    args = parser.parse_args()
    data = np.genfromtxt(args.data_path, delimiter=",").astype(np.float32)
    start = time.time()
    c, num_clust, req_c = FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True)
    print('Time Elapsed: {:2.2f} seconds'.format(time.time() - start))

    # Write back
    if args.output_path is not None:
        print('Writing back the results on the provided path ...')
        np.savetxt(args.output_path + '/c.csv', c, delimiter=',', fmt='%d')
        np.savetxt(args.output_path + '/num_clust.csv', np.array(num_clust), delimiter=',', fmt='%d')
        if req_c is not None:
            np.savetxt(args.output_path + '/req_c.csv', req_c, delimiter=',', fmt='%d')
    else:
        print('Results are not written back as the --output-path was not provided')


if __name__ == '__main__':
    main()