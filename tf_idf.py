import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data from file
file_path = 'twitter_dataset/nodes.csv'
tweets_df = pd.read_csv(file_path)

# Get tweets and labels from dataframe
tweets = tweets_df['text_cleaned'].tolist()
labels = tweets_df['text_label'].tolist()

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data for all tweets
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
tfidf_dense = tfidf_matrix.todense()
terms = tfidf_vectorizer.get_feature_names_out()

# Output file path
output_file = 'output_twitter/tf_idf_scores_each_node.txt'
with open(output_file, 'w') as f_out:
    # Iterate through each tweet
    for i, (tweet, label) in enumerate(zip(tweets, labels)):
        tfidf_scores = list(zip(terms, tfidf_dense[i].tolist()[0]))
        sorted_tfidf = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        
        # Write TF-IDF scores for the tweet to the output file
        f_out.write(f"TF-IDF scores for tweet {i+1} (label '{label}'):\n")
        for term, score in sorted_tfidf[:10]:
            f_out.write(f"{term}: {score:.8f}\n")
        f_out.write("\n")
