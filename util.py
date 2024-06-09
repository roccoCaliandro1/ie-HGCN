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



def load_imdb3228(train_percent):
	current_dir = os.path.dirname(os.path.abspath(__file__))
	hgcn_path = os.path.join(current_dir, 'data', 'imdb3228', 'imdb3228_hgcn_' + 
				str(train_percent)+'.pkl')
	print('hgcn load: ', hgcn_path, '\n')
	with open(hgcn_path, 'rb') as in_file:
		(label, ft_dict, adj_dict) = pickle.load(in_file)
		adj_dict['m']['a'] = adj_dict['m']['a'].to_sparse()
		adj_dict['m']['u'] = adj_dict['m']['u'].to_sparse()
		adj_dict['m']['d'] = adj_dict['m']['d'].to_sparse()
		
		adj_dict['a']['m'] = adj_dict['a']['m'].to_sparse()
		adj_dict['u']['m'] = adj_dict['u']['m'].to_sparse()
		adj_dict['d']['m'] = adj_dict['d']['m'].to_sparse()

	return label, ft_dict, adj_dict




if __name__ == '__main__':
	load_imdb3228(0.2)
