import base64
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the cyberbullying dataset (replace with actual path)
file_path = os.path.join(os.path.dirname(__file__), 'cyberbullying_tweets.csv')
data = pd.read_csv(file_path)

# Ensure the dataset contains the required columns
if 'tweet_text' not in df.columns or 'cyberbullying_type' not in df.columns:
    raise ValueError("Dataset must contain 'tweet_text' and 'cyberbullying_type' columns.")

# Create a dictionary mapping each tweet to its bullying type
tweet_bullying_mapping = dict(zip(df['tweet_text'], df['cyberbullying_type']))

# Preprocessing and classification function
def preprocess_and_classify(tweets):
    if len(tweets) < 3:
        raise ValueError("At least 3 tweets are required for clustering.")

    # Vectorize tweets using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(tweets)

    # Compute sentiment polarity using TextBlob
    sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]

    # Compute tweet length (number of words)
    tweet_lengths = [len(tweet.split()) for tweet in tweets]

    # Combine features into one matrix
    features = np.column_stack([X.toarray(), tweet_lengths, sentiments])

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    labels = kmeans.labels_

    # DBSCAN anomaly detection
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(features)
    anomalies = [i for i, label in enumerate(dbscan_labels) if label == -1]

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(features)

    # Determine bullying type from the dataset mapping
    bullying_types = []
    for tweet in tweets:
        bullying_type = tweet_bullying_mapping.get(tweet, "Other or Non-Bullying")
        bullying_types.append(bullying_type)

    return labels, anomalies, sentiments, tweet_lengths, X_reduced, bullying_types

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet_input = request.form.get('tweet')
    if not tweet_input:
        return render_template('index.html', error="No tweet input received.")
    
    tweet_list = tweet_input.splitlines()

    try:
        labels, anomalies, sentiments, tweet_lengths, X_reduced, bullying_types = preprocess_and_classify(tweet_list)
    except ValueError as e:
        return render_template('index.html', error=str(e))

    result = {
        'labels': labels.tolist(),
        'anomalies': anomalies,
        'sentiments': sentiments,
        'tweet_lengths': tweet_lengths,
        'bullying_types': bullying_types,
        'user_tweet': tweet_input,
        'numbered_tweets': [(i+1, tweet) for i, tweet in enumerate(tweet_list)],
    }

    # Generate word cloud for Cluster 0
    cluster_data = [tweet_list[i] for i in range(len(tweet_list)) if labels[i] == 0]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(cluster_data))

    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    wordcloud_img = base64.b64encode(img.getvalue()).decode()

    # Generate PCA plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('PCA Plot of Clusters')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    pca_img_base64 = base64.b64encode(img.getvalue()).decode()

    result['wordcloud_img'] = wordcloud_img
    result['pca_img_base64'] = pca_img_base64

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
