import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import pickle
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

def load_twitter():
    path='./twitter_dataset/'
    dataset='twitter'
	
    # load the movie_feature (m_ft) pickle, created during the preprocessing phase
    with open('{}{}_movie_feature.pkl'.format(path, dataset), 'rb') as in_file:
        m_ft = pickle.load(in_file)

	# load the adjacency matrices, created during the preprocessing phase
    with open('{}{}_sp_adj_mats.pkl'.format(path, dataset), 'rb') as in_file:
        (sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g) = pickle.load(in_file)


    print("Ciao")


if __name__ == '__main__':
	load_twitter()	
