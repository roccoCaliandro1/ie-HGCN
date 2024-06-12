import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import pickle
import torch
import os
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer



def row_normalize(mat):
    """Row-normalize matrix"""
    rowsum = mat.sum(1)
    rowsum[rowsum == 0.] = 0.01
    return mat / rowsum



def sp_coo_2_sp_tensor(sp_coo_mat):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def load_twitter(network_type, dim):
	current_dir = os.path.dirname(os.path.abspath(__file__))
	twitter_path = os.path.join(current_dir, 'twitter_dataset','twitter_hgcn_'+str(network_type)+'_'+str(dim)+'.pkl')
	print('twitter load: ', twitter_path, '\n')
	with open(twitter_path, 'rb') as in_file:
		(label, uu_dict, adj_dict_uu) = pickle.load(in_file)
		adj_dict_uu['u']['u'] = adj_dict_uu['u']['u'].to_sparse()

	return label, uu_dict, adj_dict_uu