from twitter_clean import launch_twitter_clean
from twitter_util_preprocessing import load_twitter
from twitter_train import exec_train

if __name__ == '__main__':
    # We are going here to launch the first twitter preprocessing
    launch_twitter_clean('social')
    #launch_twitter_clean('spatial')

    # We are going here to launch the second twitter preprocessing
    load_twitter('social', 128)
    load_twitter('social', 256)
    load_twitter('social', 512)
    load_twitter('social', 768)

    #load_twitter('spatial', 128)
    #load_twitter('spatial', 256)
    #load_twitter('spatial', 512)
    #load_twitter('spatial', 768)
    
    # We are going here to launch the twitter train and collect the results
    exec_train('social', 128)  
    exec_train('social', 256)
    exec_train('social', 512)
    exec_train('social', 768)