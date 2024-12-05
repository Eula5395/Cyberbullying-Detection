from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import numpy as np

# Initialize models
vectorizer = TfidfVectorizer(stop_words='english')
kmeans = KMeans(n_clusters=3, random_state=42)
dbscan = DBSCAN(eps=0.5, min_samples=2)
scaler = StandardScaler()

def preprocess_and_cluster(tweets, fit_vectorizer=False):
    if fit_vectorizer:
        # Fit the vectorizer only once when fit_vectorizer is True
        X = vectorizer.fit_transform(tweets)
    else:
        # Use the already fitted vectorizer
        X = vectorizer.transform(tweets)

    # Compute sentiment polarity using TextBlob
    sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]

    # Compute tweet length
    tweet_lengths = [len(tweet.split()) for tweet in tweets]

    # Combine features into one matrix
    features = np.column_stack([X.toarray(), tweet_lengths, sentiments])

    # Standardize features (important for clustering)
    features = scaler.fit_transform(features)

    # KMeans clustering
    kmeans.fit(features)
    labels = kmeans.labels_

    # DBSCAN anomaly detection
    dbscan_labels = dbscan.fit_predict(features)
    anomalies = [i for i, label in enumerate(dbscan_labels) if label == -1]

    return labels, anomalies, sentiments, tweet_lengths
