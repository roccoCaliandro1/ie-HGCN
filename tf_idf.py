import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter

# Load data from file
file_path = 'twitter_dataset/nodes.csv'
tweets_df = pd.read_csv(file_path)

# Get tweets and labels from dataframe
tweets = tweets_df['text_cleaned'].tolist()
labels = tweets_df['text_label'].tolist()

# Initialization
tfidf_vectorizer = TfidfVectorizer()
tfidf_scores_by_label = defaultdict(list)

# Aggregate tweets by label
for i, label in enumerate(labels):
    tfidf_scores_by_label[label].append(tweets[i])

# Identify common terms across all labels
all_terms = []
for label, tweets_list in tfidf_scores_by_label.items():
    label_documents = ' '.join(tweets_list)
    tfidf_vectorizer.fit([label_documents])
    terms = tfidf_vectorizer.get_feature_names_out()
    all_terms.extend(terms)

# Find terms that appear in more than one label
term_counts = Counter(all_terms)
common_terms = {term for term, count in term_counts.items() if count > 1}

# Output file path
output_file = 'output_twitter/tf_idf_scores_all_labels.txt'
with open(output_file, 'w') as f_out:
    # Compute TF-IDF scores for each label
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

        # Remove terms that appear in more than one label
        filtered_tfidf = [(term, score) for term, score in sorted_tfidf if term not in common_terms]

        # Write TF-IDF scores for the label to the output file
        f_out.write(f"TF-IDF scores for label '{label}':\n\n")
        for term, score in filtered_tfidf[:100]:
            f_out.write(f"{term}: {score:.8f}\n")
        f_out.write("\n")
