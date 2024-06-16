from twitter_clean import launch_twitter_clean
from twitter_util_preprocessing import load_twitter
from twitter_train import exec_train
from word2vec_train import trainWord2Vec
from compute_embeddings import computeW2Vembeddings, twitter4SSEEmbeddings

if __name__ == '__main__':
    trainW2v = False
    computeW2vEmbeddings = True
    cleaning = False
    loadPklCompute = False
    trainModel = False

    #train the word2vec model
    if trainW2v:
        trainWord2Vec(128)
        trainWord2Vec(256)
        trainWord2Vec(512)
    
    #compute the word2vec and the Twitter4SSE embeddings
    if computeW2vEmbeddings:
        computeW2Vembeddings(128)
        computeW2Vembeddings(256)
        computeW2Vembeddings(512)
        twitter4SSEEmbeddings()

    # We are going here to launch the first twitter preprocessing
    if cleaning:
        launch_twitter_clean('no')
        launch_twitter_clean('social')
        launch_twitter_clean('spatial', 0)
        launch_twitter_clean('spatial', 0.49)
        launch_twitter_clean('spatial', 0.69)

    # We are going here to launch the second twitter preprocessing
    if loadPklCompute:
        load_twitter('no', 128)
        load_twitter('no', 256)
        load_twitter('no', 512)
        load_twitter('no', 768)

        load_twitter('social', 128)
        load_twitter('social', 256)
        load_twitter('social', 512)
        load_twitter('social', 768)

        load_twitter('spatial', 128, 0)
        load_twitter('spatial', 256, 0)
        load_twitter('spatial', 512, 0)
        load_twitter('spatial', 768, 0)

        load_twitter('spatial', 128, 0.49)
        load_twitter('spatial', 256, 0.49)
        load_twitter('spatial', 512, 0.49)
        load_twitter('spatial', 768, 0.49)

        load_twitter('spatial', 128, 0.69)
        load_twitter('spatial', 256, 0.69)
        load_twitter('spatial', 512, 0.69)
        load_twitter('spatial', 768, 0.69)

     # We are going here to launch the twitter train and collect the results
    if trainModel:
       
        exec_train('no', 128)
        exec_train('no', 256)
        exec_train('no', 512)
        exec_train('no', 768)

        exec_train('social', 128)  
        exec_train('social', 256)
        exec_train('social', 512)
        exec_train('social', 768)

        exec_train('spatial', 128, 0)  
        exec_train('spatial', 256, 0)
        exec_train('spatial', 512, 0)
        exec_train('spatial', 768, 0)

        exec_train('spatial', 128, 0.49)  
        exec_train('spatial', 256, 0.49)
        exec_train('spatial', 512, 0.49)
        exec_train('spatial', 768, 0.49)

        exec_train('spatial', 128, 0.69)  
        exec_train('spatial', 256, 0.69)
        exec_train('spatial', 512, 0.69)
        exec_train('spatial', 768, 0.69)