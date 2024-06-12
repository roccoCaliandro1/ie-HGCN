import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import pickle
import torch
import os
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

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
    path='./twitter_dataset/'
	
    # load the train embeddings file, created during the preprocessing phase
    with open('{}train_embeddings_'+str(dim)+'.pkl'.format(path), 'rb') as in_file:
        u_ft_train = pickle.load(in_file)
        

    # load the test embeddings file, created during the preprocessing phase
    with open('{}test_embeddings_'+str(dim)+'.pkl'.format(path), 'rb') as in_file:
        u_ft_test = pickle.load(in_file)

    #join the train and test embeddings along with thier userIds    
    full_features = np.column_stack((
         np.concatenate((np.asarray(u_ft_train['userId']) , np.asarray(u_ft_test['userId']))), 
         np.concatenate((np.asarray(u_ft_train['embeddings']) , np.asarray(u_ft_test['embeddings']))
                        )))
    
    #sort the full features based on userIds
    full_features = full_features[full_features[:,0].argsort()]

    #remove the userIds from the full features, now we have the ordered embeddings
    full_features = full_features[:,1:]

	# load the adjacency matrices, created during the preprocessing phase
    with open('{}twitter_sp_uu_'+str(network_type)+'_adj_mats.pkl'.format(path), 'rb') as in_file:
        (sp_A_uu_sn) = pickle.load(in_file)

    A_uu_sn = sp_A_uu_sn.tocsr()

    # m_label contains only values between 0 and 3 that are the indices of the genres related to the movies
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_label_path = os.path.join(current_dir, 'twitter_dataset', 'full_label.csv')
    full_label = np.genfromtxt(full_label_path, delimiter=',', dtype=np.int32)

    full_label_sorted = np.sort(full_label, axis=0)

    label = {}
    u_label = torch.LongTensor(full_label_sorted[:,1])
    idx_train_u = torch.LongTensor(u_ft_train['userId'])
    idx_test_u = torch.LongTensor(u_ft_test['userId'])
    label['u'] = [u_label, idx_train_u, idx_test_u]

    ft_dict = {}
    u_ft_std = (full_features - full_features.mean(0)) / full_features.std(0)
    # ft_dict['u'] = torch.FloatTensor(full_features) da provare senza z_score
    ft_dict['u'] = torch.FloatTensor(u_ft_std)

    adj_dict = {'u':{}}
    adj_dict['u']['u'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(A_uu_sn)))
    # adj_dict['a']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(A_m_a.transpose())))

	# hgcn write
	# Save Processed Data for Heterogeneous Graph Convolutional Networks (HGCN):
    hgcn_path = './twitter_dataset/twitter_hgcn_'+str(network_type)+'_'+str(dim)+'.pkl'
    print('hgcn dump: ', hgcn_path)
    with open(hgcn_path, 'wb') as out_file:
        adj_dict['u']['u'] = adj_dict['u']['u'].to_dense()
     
        pickle.dump((label, ft_dict, adj_dict), out_file)