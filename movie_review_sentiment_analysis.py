import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from gensim.models import Word2Vec
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA
import seaborn as sns
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Load the IMDB dataset from NLTK
def load_imdb_dataset():
    reviews = []
    for fileid in movie_reviews.fileids():
        category = movie_reviews.categories(fileid)[0]
        review = ' '.join(movie_reviews.words(fileid))
        reviews.append((review, category))
    return reviews

# Preprocess the reviews
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

# Load and preprocess the dataset
reviews = load_imdb_dataset()
texts, labels = zip(*reviews)
tokenized_reviews = [preprocess_text(review) for review in tqdm(texts, desc="Tokenizing and Preprocessing Reviews")]

# Train the Word2Vec model
model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=5, sg=0)
print("Gensim and Word2Vec parts are working fine.")

# Function to calculate positivity and negativity
def calculate_sentiment_scores(reviews):
    positive_counts = []
    negative_counts = []
    for review in tqdm(reviews, desc="Calculating Sentiment Scores"):
        positive_count = sum(1 for word in review if TextBlob(word).sentiment.polarity > 0)
        negative_count = sum(1 for word in review if TextBlob(word).sentiment.polarity < 0)
        total_words = len(review)
        positive_counts.append(positive_count / total_words if total_words > 0 else 0)
        negative_counts.append(negative_count / total_words if total_words > 0 else 0)
    return positive_counts, negative_counts

# Calculate positivity and negativity scores
positivity_scores, negativity_scores = calculate_sentiment_scores(tokenized_reviews)
mean_positivity = np.mean(positivity_scores)
std_positivity = np.std(positivity_scores)
mean_negativity = np.mean(negativity_scores)
std_negativity = np.std(negativity_scores)

# Compute the normalized sentiment score for each review
def compute_normalized_score(pos, neg, mean_pos, std_pos, mean_neg, std_neg):
    return (pos - mean_pos) / std_pos - (neg - mean_neg) / std_neg

normalized_scores = [compute_normalized_score(pos, neg, mean_positivity, std_positivity, mean_negativity, std_negativity)
                     for pos, neg in zip(positivity_scores, negativity_scores)]

# Function to determine the color based on normalized sentiment score
def sentiment_to_color(score):
    if score > 0:
        return 'red'
    elif score < 0:
        return 'blue'
    else:
        return 'gray'

# Apply color coding
colors = [sentiment_to_color(score) for score in normalized_scores]

# Plot sentiment scores with colors
plt.figure(figsize=(10, 6))
plt.scatter(range(len(normalized_scores)), normalized_scores, c=colors)
plt.title('Sentiment Analysis of IMDB Reviews')
plt.xlabel('Review Index')
plt.ylabel('Normalized Sentiment Score')
plt.show()

# Finding similar words
similar_words = model.wv.most_similar("movie")
print("Words similar to 'movie':", similar_words)

# Getting the vector for a word
vector = model.wv["film"]
print("Vector for 'film':", vector)

# Evaluating word analogies
try:
    analogy = model.wv.most_similar(positive=['good', 'movie'], negative=['bad'])
    print("Analogies (good + movie - bad):", analogy)
except KeyError as e:
    print(f"Word not found in vocabulary: {e}")

# Advanced Visualization: Word Embedding Clustering
def plot_word_embeddings(model, words, top_n=10):
    word_vectors = np.array([model.wv[word] for word in words if word in model.wv][:top_n])
    pca = PCA(n_components=2)
    word_vectors_2d = pca.fit_transform(word_vectors)
    plt.figure(figsize=(10, 10))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c='orange')
    for i, word in enumerate(words[:top_n]):
        plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))
    plt.title('PCA of Word Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Visualizing the top similar words to "movie"
plot_word_embeddings(model, [word for word, score in similar_words], top_n=10)

# Additional EDA: Word Frequency Analysis
all_words = [word for review in tokenized_reviews for word in review]
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(20)

# Plotting the most common words
words, counts = zip(*most_common_words)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(words), y=list(counts))
plt.title('Most Common Words in IMDB Reviews')
plt.xlabel('Words')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()

# Save the trained Word2Vec model for future use
model.save("word2vec_imdb.model")

# Load the model (demonstrating how to load the saved model)
loaded_model = Word2Vec.load("word2vec_imdb.model")
print("Loaded Word2Vec model:", loaded_model)

# Function to predict sentiment of new reviews
def predict_sentiment(review, model):
    tokenized_review = preprocess_text(review)
    positive_count = sum(1 for word in tokenized_review if TextBlob(word).sentiment.polarity > 0)
    negative_count = sum(1 for word in tokenized_review if TextBlob(word).sentiment.polarity < 0)
    total_words = len(tokenized_review)
    if total_words == 0:
        return "Neutral"
    normalized_score = compute_normalized_score(positive_count / total_words, negative_count / total_words,
                                                mean_positivity, std_positivity, mean_negativity, std_negativity)
    if normalized_score > 0:
        return "Positive"
    elif normalized_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Example prediction
example_review = "The movie was fantastic with excellent performances."
predicted_sentiment = predict_sentiment(example_review, loaded_model)
print(f"The sentiment of the example review is: {predicted_sentiment}")

# Additional Feature: Sentiment Distribution Plot
def plot_sentiment_distribution(normalized_scores):
    plt.figure(figsize=(10, 6))
    sns.histplot(normalized_scores, bins=30, kde=True)
    plt.title('Distribution of Normalized Sentiment Scores')
    plt.xlabel('Normalized Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

plot_sentiment_distribution(normalized_scores)

# Additional Feature: Word Cloud for Positive and Negative Words
def plot_wordcloud(data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

positive_words = [word for review in tokenized_reviews for word in review if TextBlob(word).sentiment.polarity > 0]
negative_words = [word for review in tokenized_reviews for word in review if TextBlob(word).sentiment.polarity < 0]

plot_wordcloud(positive_words, 'Positive Words WordCloud')
plot_wordcloud(negative_words, 'Negative Words WordCloud')

# Additional Feature: Top Positive and Negative Reviews
def find_extreme_reviews(texts, scores, num_reviews=5):
    sorted_indices = np.argsort(scores)
    most_negative = sorted_indices[:num_reviews]
    most_positive = sorted_indices[-num_reviews:][::-1]
    print("\nMost Negative Reviews:")
    for idx in most_negative:
        print(f"Score: {scores[idx]}, Review: {texts[idx][:500]}...")
    print("\nMost Positive Reviews:")
    for idx in most_positive:
        print(f"Score: {scores[idx]}, Review: {texts[idx][:500]}...")

find_extreme_reviews(texts, normalized_scores)
