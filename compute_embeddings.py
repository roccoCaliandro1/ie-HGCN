import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def twitter4SSEEmbeddings():
    do_train_pkl = True
    do_test_pkl = True
    if(do_train_pkl):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(current_dir, 'twitter_dataset', 'train.csv')
        train = pd.read_csv(train_path, sep=',', index_col=0).drop('index',axis=1).drop('label',axis=1)
        model = SentenceTransformer('digio/Twitter4SSE')
        embeddings = model.encode(train['text_cleaned'])
        path = os.path.join(current_dir, 'twitter_dataset', 'embedded_datasets', '768_dim', 'train_embeddings_768' + '.pkl')
        
        with open(path, "wb") as fOut:
            pickle.dump({'userId': train['id'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
    if(do_test_pkl):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_path = os.path.join(current_dir, 'twitter_dataset', 'test.csv')
        test = pd.read_csv(test_path, sep=',', index_col=0).drop('index',axis=1).drop('label',axis=1)
        model = SentenceTransformer('digio/Twitter4SSE')
        embeddings = model.encode(test['text_cleaned'])
        path = os.path.join(current_dir, 'twitter_dataset', 'embedded_datasets', '768_dim', 'test_embeddings_768' + '.pkl')
        
        with open(path, "wb") as fOut:
            pickle.dump({'userId': test['id'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

def computeW2Vembeddings(dim):
    do_train_pkl = True
    do_test_pkl = True
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(current_dir, 'twitter_dataset', 'train.csv')
    test_path = os.path.join(current_dir, 'twitter_dataset', 'test.csv')
    train = pd.read_csv(train_path, sep=',', index_col=0).drop('index',axis=1).drop('label',axis=1)
    test = pd.read_csv(test_path, sep=',', index_col=0).drop('index',axis=1).drop('label',axis=1)

    # rifare lista per train e per test
    train_list = []
    train_vect = train['text_cleaned']


    for s in train_vect:
        train_list.append(s.split(" "))

    test_list = []
    test_vect = test['text_cleaned']

    for s in test_vect:
        test_list.append(s.split(" "))

    w2v_path = os.path.join(current_dir, 'twitter_dataset', 'word2vec_models', 'word2vec_model_' + str(dim) + '.pkl')
    
    with open(w2v_path, "rb") as in_file:
        w2v_model = pickle.load(in_file)
    if do_train_pkl:
        ##TRAIN#####################################################################
        print('Embedding training set in '+str(dim)+' dimensions.')
        node_train_embeddings = []
        # per train cicliamo nella lista delle parole, embedding di tutte le parole e accumuliamo in un vettore
        i = 0
        for sentence in train_list:
            #inizializzazione
            node_embedding = np.zeros(dim)

            for word in sentence:
                #calcoliamo valore per quella parola
                if word in list(w2v_model.wv.index_to_key):
                    vector = w2v_model.wv[word]
                    # sommiamo
                    node_embedding += vector

            node_train_embeddings.append(node_embedding)
            i += 1
            print(i)

        embeddings = np.asarray(node_train_embeddings)

        path = os.path.join(current_dir, 'twitter_dataset', 'embedded_datasets', str(dim)+'_dim', 'train_embeddings_' + str(dim) + '.pkl')
        with open(path, "wb") as fOut:
            pickle.dump({'userId': train['id'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        ############################################################################

    if do_test_pkl:
        ##TEST######################################################################
        print('Embedding test set in '+str(dim)+' dimensions.')
        node_test_embeddings = []
        # per test cicliamo nella lista delle parole, embedding di tutte le parole e accumuliamo in un vettore
        i = 0
        for sentence in test_list:
            #inizializzazione
            node_embedding = np.zeros(dim)

            for word in sentence:
                #calcoliamo valore per quella parola
                if word in list(w2v_model.wv.index_to_key):
                    vector = w2v_model.wv[word]
                    # sommiamo
                    node_embedding += vector

            node_test_embeddings.append(node_embedding)
            i += 1
            print(i)

        embeddings = np.asarray(node_test_embeddings)

        path = os.path.join(current_dir, 'twitter_dataset', 'embedded_datasets', str(dim)+'_dim', 'test_embeddings_' + str(dim) + '.pkl')
        
        with open(path, "wb") as fOut:
            pickle.dump({'userId': test['id'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)