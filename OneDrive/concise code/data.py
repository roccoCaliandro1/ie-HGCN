import numpy as np
import scipy.sparse as sp
import torch
import pickle



def sp_coo_2_sp_tensor(sp_coo_mat):
    '''Convert a scipy sparse matrix to a torch sparse tensor.'''
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def train_val_test_split(label_shape, train_percent):
    rand_idx = np.random.permutation(label_shape)
    val_percent =  (1.0 - train_percent)/2
    idx_train = torch.LongTensor(rand_idx[int(label_shape*0.0): int(label_shape*train_percent)])
    idx_val = torch.LongTensor(rand_idx[int(label_shape*train_percent): int(label_shape*(train_percent + val_percent))])
    idx_test = torch.LongTensor(rand_idx[int(label_shape*(train_percent + val_percent)): int(label_shape*1.0)])
    return idx_train, idx_val, idx_test


def load_imdb_3228(train_percent):

    hgcn_path = './datasets/imdb_3228/imdb3228_hgcn_'+str(0.2)+'.pkl'
    print('load data from: ', hgcn_path, '\n')
    with open(hgcn_path, 'rb') as in_file:
        (label, ft_dict, adj_dict) = pickle.load(in_file)

        m_label = label['m'][0]
        idx_train_m, idx_val_m, idx_test_m = train_val_test_split(m_label.shape[0], train_percent)
        label['m'] = [m_label, idx_train_m, idx_val_m, idx_test_m]

        adj_dict['m']['a'] = adj_dict['m']['a'].to_sparse()
        adj_dict['m']['u'] = adj_dict['m']['u'].to_sparse()
        adj_dict['m']['d'] = adj_dict['m']['d'].to_sparse()
        
        adj_dict['a']['m'] = adj_dict['a']['m'].to_sparse()
        adj_dict['u']['m'] = adj_dict['u']['m'].to_sparse()
        adj_dict['d']['m'] = adj_dict['d']['m'].to_sparse()


    return label, ft_dict, adj_dict



if __name__ == '__main__':

    load_imdb_3228(0.01)
    # load_imdb_3228(0.05)
    # load_imdb_3228(0.1)
    # load_imdb_3228(0.2)
    # load_imdb_3228(0.4)
    # load_imdb_3228(0.6)
    # load_imdb_3228(0.8)

    print('Successfully Loaded')