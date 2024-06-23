import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the CSV file
file_path = 'twitter_dataset/nodes.csv'
tweets_df = pd.read_csv(file_path)

# Extract the tweets from the 'text_cleaned' column
tweets = tweets_df['text_cleaned'].tolist()
labels = tweets_df['label'].tolist()
unique_labels = set(labels)

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)

# Convert the matrix to a dense array and get feature names
tfidf_dense = tfidf_matrix.todense()
terms = tfidf_vectorizer.get_feature_names_out()

# Display the TF-IDF scores in a more readable format
tfidf_scores = []

for i, tweet in enumerate(tweets):
    tweet_tfidf = zip(terms, tfidf_dense[i].tolist()[0])
    sorted_tfidf = sorted(tweet_tfidf, key=lambda x: x[1], reverse=True)
    tweet_scores = {term: score for term, score in sorted_tfidf[:10] if score > 0}
    tfidf_scores.append(tweet_scores)

with open('output_twitter/tf_idf_out.txt', 'w') as f:
    # Print the TF-IDF scores for all tweets, only the top 10 terms
    for i, scores in enumerate(tfidf_scores):
        f.write(f"Tweet {i+1}:\n")
        for term, score in scores.items():
            f.write(f"  {term}: {score:.4f}\n")
        f.write("\n")
