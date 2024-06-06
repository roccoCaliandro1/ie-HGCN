import numpy as np
import pickle
import scipy.sparse as sp
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
twitter_file_path = os.path.join(current_dir, 'cleaned', 'small','tweets.txt')
m_raw_id_feature = np.genfromtxt(twitter_file_path, comments='#', delimiter=",", missing_values='\\N', filling_values=0.0, dtype=np.float32)
print('twitter id feature shape: ', m_raw_id_feature.shape)
m_raw_ids = m_raw_id_feature[:,0].astype(np.int32)
m_num = len(np.unique(m_raw_ids))
if m_num == len(m_raw_ids):
	print('number of unique tweets: ', m_num, '\n')
	m_id_raw2int = dict(zip(m_raw_ids, list(range(m_num))))
	m_ft = m_raw_id_feature[:,1:]
else:
	print('duplicate tweet id feature!\n')
	exit()



social_network_test_path = os.path.join(current_dir, 'cleaned', 'small', 'graph', 'social_network_test.txt')
s_n_test = np.genfromtxt(social_network_test_path, delimiter='\t', dtype=np.int32)
print('social network test shape: ', s_n_test.shape)
sn_test_raw_ids = s_n_test[:,1]
sn_unique_raw_test_ids = np.unique(sn_test_raw_ids)
sn_test_id_raw2int = dict(zip(sn_unique_raw_test_ids, list(range(len(sn_unique_raw_test_ids)))))
sn_test_num = len(sn_unique_raw_test_ids)
print('number of unique test: ', sn_test_num, '\n')
row = [sn_test_id_raw2int[sn_id] for sn_id in s_n_test[:,0].astype(np.int32)]
col = [sn_test_id_raw2int[a_id] for a_id in s_n_test[:,1]]
data = s_n_test[:,2].astype(np.float32)
sn_A = sp.coo_matrix((data, (row, col)), shape=(sn_test_num, sn_test_num))



'''
social_network_train_path = os.path.join(current_dir, 'cleaned', 'small', 'graph', 'social_network_train.txt')
s_n_train = np.genfromtxt(social_network_train_path, delimiter='\t', dtype=np.int32)
print('social network train shape: ', s_n_train.shape)
sn_train_raw_ids = s_n_train[:,1]
sn_unique_raw_train_ids = np.unique(sn_train_raw_ids)
sn_train_id_raw2int = dict(zip(sn_unique_raw_train_ids, list(range(len(sn_unique_raw_train_ids)))))
sn_train_num = len(sn_unique_raw_train_ids)
print('number of unique train: ', sn_train_num, '\n')
row = [sn_train_id_raw2int[sn_id] for sn_id in s_n_train[:,0].astype(np.int32)]
col = [sn_train_id_raw2int[a_id] for a_id in s_n_train[:,1]]
data = s_n_train[:,2].astype(np.float32)
sn_A = sp.coo_matrix((data, (row, col)), shape=(sn_train_num, sn_train_num))
'''