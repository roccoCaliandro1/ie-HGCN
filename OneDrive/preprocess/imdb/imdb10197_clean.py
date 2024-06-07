import numpy as np
import pickle
import scipy.sparse as sp
import os

# load data from txt
current_dir = os.path.dirname(os.path.abspath(__file__))
movie_file_path = os.path.join(current_dir, 'cleaned', 'movie.txt')
m_raw_id_feature = np.genfromtxt(movie_file_path, delimiter='\t', missing_values='\\N', filling_values=0.0, dtype=np.float32)
print('movie id feature shape: ', m_raw_id_feature.shape)
m_raw_ids = m_raw_id_feature[:,0].astype(np.int32)
m_num = len(np.unique(m_raw_ids))
# check if distinct ids are equal to total ids number
if m_num == len(m_raw_ids):
	print('number of unique movie: ', m_num, '\n')
	# dictionary movie_id, sequential_id
	m_id_raw2int = dict(zip(m_raw_ids, list(range(m_num))))
	# we remove here the first column
	m_ft = m_raw_id_feature[:,1:]
else:
	print('duplicate paper id feature!\n')
	exit()

# load data from txt
movie_actor_path = os.path.join(current_dir, 'cleaned', 'movie_actor.txt')
m_a = np.genfromtxt(movie_actor_path, delimiter='\t', dtype=str)
print('movie_actor shape: ', m_a.shape)
# create ids with actorIds
a_raw_ids = m_a[:,1]
# create unique actorIds: why is contatenated with a number like 1000057-don_adams?
a_unique_raw_ids = np.unique(a_raw_ids)
# convert each id with a number from 0 to n-1
a_id_raw2int = dict(zip(a_unique_raw_ids, list(range(len(a_unique_raw_ids)))))
# get the number of distinct actors
a_num = len(a_unique_raw_ids)
print('number of unique actor: ', a_num, '\n')
# creation of: row, col and data for the creation of a sparse matrix
row = [m_id_raw2int[m_id] for m_id in m_a[:,0].astype(np.int32)]
col = [a_id_raw2int[a_id] for a_id in m_a[:,1]]
data = m_a[:,2].astype(np.float32)
sp_A_m_a = sp.coo_matrix((data, (row, col)), shape=(m_num, a_num))

# read the txt movie_country containing film Id, Country
movie_country_path = os.path.join(current_dir, 'cleaned', 'movie_country.txt')
m_c = np.genfromtxt(movie_country_path, delimiter='\t', dtype=str)
# print the size of movie_country
print('movie_country shape: ', m_c.shape)
# create ids of Countries
c_raw_ids = m_c[:,1]
# creation of unique ids of Countries
c_unique_raw_ids = np.unique(c_raw_ids)
# convert each id with a number from 0 to n-1
c_id_raw2int = dict(zip(c_unique_raw_ids, list(range(len(c_unique_raw_ids)))))
c_num = len(c_unique_raw_ids)
# prints number of distinct countries
print('number of unique country: ', c_num, '\n')
# creation of: row, col and data for the creation of a sparse matrix
row = [m_id_raw2int[m_id] for m_id in m_c[:,0].astype(np.int32)]
col = [c_id_raw2int[c_id] for c_id in m_c[:,1]]
data = np.ones(m_c.shape[0], dtype=np.float32)
sp_A_m_c = sp.coo_matrix((data, (row, col)), shape=(m_num, c_num))


# read the txt movie_director containing film Id, director
movie_director_path = os.path.join(current_dir, 'cleaned', 'movie_director.txt')
m_d = np.genfromtxt(movie_director_path, delimiter='\t', dtype=str)
# print the size of movie_director
print('movie_director shape: ', m_d.shape)
# create ids of directors
d_raw_ids = m_d[:,1]
# creation of unique ids of directors
d_unique_raw_ids = np.unique(d_raw_ids)
# converts each id with a number from 0 to n-1
d_id_raw2int = dict(zip(d_unique_raw_ids, list(range(len(d_unique_raw_ids)))))
d_num = len(d_unique_raw_ids)
# prints number of distinct directors
print('number of unique director: ', d_num, '\n')
# creation of: row, col and data for the creation of a sparse matrix
row = [m_id_raw2int[m_id] for m_id in m_d[:,0].astype(np.int32)]
col = [d_id_raw2int[d_id] for d_id in m_d[:,1]]
data = np.ones(m_d.shape[0], dtype=np.float32)
sp_A_m_d = sp.coo_matrix((data, (row, col)), shape=(m_num, d_num))


# read the txt movie_tag containing: movieID, tagID, tagWeight
movie_tag_path = os.path.join(current_dir, 'cleaned', 'movie_tag.txt')
m_t = np.genfromtxt(movie_tag_path, delimiter='\t', dtype=np.int32)
# print the size of movie_tag
print('movie_tag shape: ', m_t.shape)
# create ids of tagIds
t_raw_ids = m_t[:,1]
# creation of unique ids
t_unique_raw_ids = np.unique(t_raw_ids)
# convert each id with a number from 0 to n-1
t_id_raw2int = dict(zip(t_unique_raw_ids, list(range(len(t_unique_raw_ids)))))
t_num = len(t_unique_raw_ids)
# prints the number of distinct tags
print('number of unique tag: ', t_num, '\n')
# creation of: row, col and data for the creation of a sparse matrix
row = [m_id_raw2int[m_id] for m_id in m_t[:,0]]
col = [t_id_raw2int[t_id] for t_id in m_t[:,1]]
# data conains tagWeight
data = m_t[:,2].astype(np.float32)
sp_A_m_t = sp.coo_matrix((data, (row, col)), shape=(m_num, t_num))



# read the txt movie_user containing movieId, user_id, rating
movie_user_path = os.path.join(current_dir, 'cleaned', 'movie_user.txt')
m_u = np.genfromtxt(movie_user_path, delimiter='\t', dtype=np.float32)
# print the size of movie_user
print('movie_user shape: ', m_u.shape)
# create ids of userIds
u_raw_ids = m_u[:,1].astype(np.int32)
# creation of unique ids
u_unique_raw_ids= np.unique(u_raw_ids)
# convert each id with a number between 0 and n-1
u_id_raw2int = dict(zip(u_unique_raw_ids, list(range(len(u_unique_raw_ids)))))
u_num = len(u_unique_raw_ids)
# prints the number of distinct userss
print('number of unique user: ', u_num, '\n')
# creation of: row, col and data for the creation of a sparse matrix
row = [m_id_raw2int[m_id] for m_id in m_u[:,0].astype(np.int32)]
col = [u_id_raw2int[u_id] for u_id in m_u[:,1].astype(np.int32)]
# data conains the rating
data = m_u[:,2]
sp_A_m_u = sp.coo_matrix((data, (row, col)), shape=(m_num, u_num))


# read the txt movie_genre containing: movieID, genre
movie_genre_path = os.path.join(current_dir, 'cleaned', 'movie_genre.txt')
m_g = np.genfromtxt(movie_genre_path, delimiter='\t', dtype=str)
# prints the size of movie_genre
print('movie_genre shape: ', m_g.shape)
# create ids of genres
g_raw_ids = m_g[:,1]
# creation of unique genres
g_unique_raw_ids = np.unique(g_raw_ids)
# convert each id with a number from 0 to n-1
g_id_raw2int = dict(zip(g_unique_raw_ids, list(range(len(g_unique_raw_ids)))))
g_num = len(g_unique_raw_ids)
# prints the number of distinct genres
print('number of unique genre: ', g_num, '\n')
# creation of: row, col and data for the creation of a sparse matrix
row = [m_id_raw2int[m_id] for m_id in m_g[:,0].astype(np.int32)]
col = [g_id_raw2int[g_id] for g_id in m_g[:,1]]
data = np.ones(m_g.shape[0], dtype=np.float32)
sp_A_m_g = sp.coo_matrix((data, (row, col)), shape=(m_num, g_num))


# save into imdb10197_ids_map_dict.pkl all the IDS converted with a number between 0 and n-1
# creaeted with dict(zip(unique_ids, list(range(len(unique_ids)))))
file_path = os.path.join(current_dir, 'dump', 'imdb10197_ids_map_dict.pkl')
with open(file_path, 'wb') as out_file:
	pickle.dump((m_id_raw2int, a_id_raw2int, c_id_raw2int, d_id_raw2int, t_id_raw2int, u_id_raw2int, g_id_raw2int), out_file)


# save into imdb10197_movie_feature.pkl ALL the features of movie id, title, year and so on...
file_path = os.path.join(current_dir, 'dump', 'imdb10197_movie_feature.pkl')
with open(file_path, 'wb') as out_file:
	pickle.dump(m_ft, out_file)


# save into imdb10197_sp_adj_mats.pkl all the sparse matrices created with col, row and data
file_path = os.path.join(current_dir, 'dump', 'imdb10197_sp_adj_mats.pkl')
with open(file_path, 'wb') as out_file:
	pickle.dump((sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g), out_file)

# From now on, we read the pickles just for check
file_path = os.path.join(current_dir, 'dump', 'imdb10197_ids_map_dict.pkl')
with open(file_path, 'rb') as in_file:
	(m_id_raw2int, a_id_raw2int, c_id_raw2int, d_id_raw2int, t_id_raw2int, u_id_raw2int, g_id_raw2int) = pickle.load(in_file)
print('number of movie: ', len(m_id_raw2int))
print('number of actor: ', len(a_id_raw2int))
print('number of country: ', len(c_id_raw2int))
print('number of director: ', len(d_id_raw2int))
print('number of tag: ', len(t_id_raw2int))
print('number of user: ', len(u_id_raw2int))
print('number of genre: ', len(g_id_raw2int))

file_path = os.path.join(current_dir, 'dump', 'imdb10197_movie_feature.pkl')
with open(file_path, 'rb') as in_file:
	m_ft = pickle.load(in_file)
print('movie feature shape: ', m_ft.shape, type(m_ft), m_ft.dtype)


file_path = os.path.join(current_dir, 'dump', 'imdb10197_sp_adj_mats.pkl')
with open(file_path, 'rb') as in_file:
	(sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g) = pickle.load(in_file)

print('m_a: ', sp_A_m_a.max(), sp_A_m_a.shape, type(sp_A_m_a))
print('m_u: ', sp_A_m_u.max(), sp_A_m_u.shape, type(sp_A_m_u))
print('m_t: ', sp_A_m_t.max(), sp_A_m_t.shape, type(sp_A_m_t))
print('m_d: ', sp_A_m_d.max(), sp_A_m_d.shape, type(sp_A_m_d))
print('m_c: ', sp_A_m_c.max(), sp_A_m_c.shape, type(sp_A_m_c))
print('m_g: ', sp_A_m_g.max(), sp_A_m_g.shape, type(sp_A_m_g))
