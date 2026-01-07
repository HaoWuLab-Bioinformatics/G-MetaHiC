from granular_ball import GranularBall
from HyperballClustering import get_radius
from splitGBs import splitGBs
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
import pandas as pd


def generate_granular_metacell(lsi_values, n_neighbors=100, max_overlap=31):

    gb_list = splitGBs(lsi_values)


    tree = KDTree(lsi_values)
    candidate_representatives = []

    for gb in gb_list:

        points_in_gb = gb.data


        center = gb.center


        _, nearest_idx = tree.query(center.reshape(1, -1), k=1)
        candidate_representatives.append(nearest_idx[0][0])


    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(lsi_values)
    nbrs_matrix = nbrs.kneighbors_graph(lsi_values).toarray()


    selected_idx = [False] * len(lsi_values)
    for idx in candidate_representatives:
        selected_idx[idx] = True


    order = np.arange(len(lsi_values))
    np.random.seed(10)
    np.random.shuffle(order)

    for idx in order:
        if selected_idx[idx]:
            continue

        selected_cells = nbrs_matrix[selected_idx]
        candidate = nbrs_matrix[idx]
        overlap = np.max(candidate.dot(selected_cells.T))

        if overlap < max_overlap:
            selected_idx[idx] = True


    metacell_assignment = nbrs_matrix[selected_idx]
    return metacell_assignment



ct = 'CD4_TCells'
lsi = pd.read_csv('../../gmetahic/datasets/atac/{}/archr_filtered_lsi.csv'.format(ct), index_col = 0)

print(lsi.shape)
lsi.index = [x.split('#')[1] for x in lsi.index]

print(len(lsi.index))


n_neighbors = 100
max_overlap = 30
metacell_assignment = generate_granular_metacell(lsi.values, n_neighbors, max_overlap)


pd.DataFrame(metacell_assignment).to_csv(f"../../gmetahic/datasets/atac/{ct}/{ct}_metacell_mask_GB_overlap{max_overlap}.csv")