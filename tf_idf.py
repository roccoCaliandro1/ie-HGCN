import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Load data from file
file_path = 'twitter_dataset/nodes.csv'
tweets_df = pd.read_csv(file_path)

# Get tweets and labels from dataframe
tweets = tweets_df['text_cleaned'].tolist()
labels = tweets_df['text_label'].tolist()

# Initialization
tfidf_vectorizer = TfidfVectorizer()
tfidf_scores_by_label = defaultdict(list)

# Aggregation of tweets by label
for i, label in enumerate(labels):
    tfidf_scores_by_label[label].append(tweets[i])

# Write in output file
output_file = 'output_twitter/tf_idf_scores_all_labels.txt'
with open(output_file, 'w') as f:
    # Computation of the score for each label
    for label, tweets_list in tfidf_scores_by_label.items():
        # Join all tweets into a single document for the label
        label_documents = [' '.join(tweets_list)]

        # Fit and transform the data for the current label
        tfidf_matrix = tfidf_vectorizer.fit_transform(label_documents)
        tfidf_dense = tfidf_matrix.todense()
        terms = tfidf_vectorizer.get_feature_names_out()

        # Get TF-IDF scores for the label
        tfidf_scores = list(zip(terms, tfidf_dense.tolist()[0]))
        sorted_tfidf = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

        # Write in output file
        f.write(f"TF-IDF scores for label '{label}':\n")
        for term, score in sorted_tfidf[:10]:
            f.write(f"{term}: {score:.4f}\n")
        f.write("\n")
