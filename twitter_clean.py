import numpy as np
import pickle
import scipy.sparse as sp
import os

import sys

def launch_twitter_clean(network_type):
	np.set_printoptions(threshold=np.inf)

	# load data from txt
	current_dir = os.path.dirname(os.path.abspath(__file__))

	# load data from txt
	sn_twitter_path = os.path.join(current_dir, 'twitter_dataset', 'full_'+ network_type + '_network.txt')
	sn_uu = np.genfromtxt(sn_twitter_path, delimiter='\t', dtype=np.int32)
	print('user_user social network shape: ', sn_uu.shape)
	# create ids with all the users by using the  entire network and by removing duplicates
	sn_uu_0 = sn_uu[:, 0]
	sn_uu_1 = sn_uu[:, 1]
	sn_uu_ids = np.array(list(set(np.concatenate([sn_uu_0, sn_uu_1]))))
	sn_uu_num = len(sn_uu_ids)

	# creation of: row, col and data for the creation of a sparse matrix
	row = sn_uu_0
	col = sn_uu_1

	# data is 1 for social network, while for spatial network is the distance
	if(network_type == 'social'):
		data = np.ones(row.shape[0], dtype=np.float32)
	elif(network_type == 'spatial'):
		data = sn_uu[:, 2].astype(np.float32)

	# Load full label in order to get tot num of users, this is needed to create the sparse matrix
	full_label_path = os.path.join(current_dir, 'twitter_dataset', 'full_label.csv')
	full_label = np.genfromtxt(full_label_path, delimiter=',', dtype=np.int32)
	sp_A_uu_sn = sp.coo_matrix((data, (row, col)), shape=(full_label.shape[0], full_label.shape[0]))

	print('number of unique users that partecipates in the '+network_type+' network: ', sn_uu_num, '\n')

	# save into twitter_sn_uu_ids.pkl all the user IDS
	file_path = os.path.join(current_dir, 'twitter_dataset', 'twitter_'+ network_type + '_uu_ids.pkl')
	with open(file_path, 'wb') as out_file:
		pickle.dump((sn_uu_ids), out_file)

	# save into twitter_sp_uu_sn_adj_mats.pkl all the sparse matrices created with col, row and data
	file_path = os.path.join(current_dir, 'twitter_dataset', 'twitter_sp_uu_' + network_type + '_adj_mats.pkl')
	with open(file_path, 'wb') as out_file:
		pickle.dump((sp_A_uu_sn), out_file)

    ###########################################################################################################
	# From now on, we read the pickles just for checking
	file_path = os.path.join(current_dir, 'twitter_dataset', 'twitter_' + network_type + '_uu_ids.pkl')
	with open(file_path, 'rb') as in_file:
		(pkl_sn_uu_ids) = pickle.load(in_file)
	print('number of users partecipating in the '+network_type+' network: ', len(pkl_sn_uu_ids))

	file_path = os.path.join(current_dir, 'twitter_dataset', 'twitter_sp_uu_'+ network_type +'_adj_mats.pkl')
	with open(file_path, 'rb') as in_file:
		(pkl_sp_A_uu_sn) = pickle.load(in_file)
	print('pkl_sp_A_uu_sn: ', pkl_sp_A_uu_sn.max(), pkl_sp_A_uu_sn.shape, type(pkl_sp_A_uu_sn))
	###########################################################################################################