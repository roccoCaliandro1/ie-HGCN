from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
# import matplotlib
# # matplotlib.use('Agg')
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def kmeans(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)



def visual_embd(seed, train_percent, embd, label):
	visual_vec = TSNE(n_components=2).fit_transform(embd)

	# markers = ['o', 'v', 's', 'X']
	# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# classes = np.unique(label).tolist()
	# for c_i in classes:
	# 	ax.scatter(visual_vec[label==c_i, 0], visual_vec[label==c_i, 1], c=colors[c_i], marker=markers[c_i], s=10)
	# plt.show()


	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(visual_vec[:,0], visual_vec[:,1], c=label, s=3)
	# plt.show()

	# fig_name = './test/visualization/visual_imdb/hgcn_tune/'+'_'+str(train_percent)+'_'+str(seed)+'.pdf'
	# fig_name = './test/visualization/visual_acm/hgcn_tune/'+'_'+str(train_percent)+'_'+str(seed)+'.pdf'
	fig_name = './test/visualization/visual_dblp/hgcn_tune/'+'_'+str(train_percent)+'_'+str(seed)+'.pdf'
	plt.savefig(fig_name)
	plt.close(fig)
