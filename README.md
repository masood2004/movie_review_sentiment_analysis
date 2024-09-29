
# Movie Review Sentiment Analysis

## Overview
This repository contains a Python-based application for analyzing sentiment in movie reviews using Natural Language Processing (NLP) techniques. It uses the IMDB dataset available in NLTK, Word2Vec for word embeddings, and TextBlob for sentiment analysis. The program preprocesses the text, trains a Word2Vec model, and performs sentiment scoring (positive and negative) on the movie reviews.

## Features
- Text Preprocessing: Tokenization, stopword removal, and text cleaning.
- Word2Vec Model: Trains a Word2Vec model on the movie reviews to create word embeddings.
- Sentiment Analysis: Calculates positivity and negativity scores for each review using TextBlob.
- Data Visualization: Visualizes sentiment analysis results using WordCloud and PCA plots.

## Installation

### Prerequisites

To run this program, ensure you have Python 3.x installed and the following libraries:

- NLTK: For loading and processing the IMDB movie reviews.
- Gensim: For Word2Vec model training.
- TextBlob: For sentiment analysis.
- Matplotlib, Seaborn: For data visualization.
- WordCloud: For creating word clouds of positive and negative sentiment words.

Install the required libraries by running:

```bash
pip install nltk gensim textblob matplotlib seaborn wordcloud tqdm
```

### Steps

1. Clone the repository:

```bash
git clone https://github.com/masood2004/movie_review_sentiment_analysis.git
cd movie-review-sentiment-analysis
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

3. Run the program:

```bash
python movie_review_sentiment_analysis.py
```


## Usage/Examples

1. Dataset: The program uses the IMDB dataset available in the NLTK corpus. It automatically downloads the required data on the first run.

2. Run the Program:

```bash
python movie_review_sentiment_analysis.py
```

This will load the dataset, preprocess the reviews, train a Word2Vec model, and compute sentiment scores for the reviews.

3. Visualization: The program generates word clouds and other visualizations to showcase positive and negative sentiment.

## Example
To analyze the IMDB movie reviews and visualize sentiment, simply run:

```bash
python movie_review_sentiment_analysis.py
```


## Dependencies

- Python 3.x
- NLTK: Natural Language Toolkit for working with the IMDB dataset.
- Gensim: For training the Word2Vec model.
- TextBlob: For sentiment analysis.
- Matplotlib, Seaborn: For visualizing data.
- WordCloud: For creating word cloud representations.

To install all dependencies:

```bash
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes.
4. Commit your changes (git commit -m 'Add some feature').
5. Push to the branch (git push origin feature-branch).
6. Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, feel free to contact:

- Name: Syed Masood Hussain
- Email: hmasood3288@gmail.com
- GitHub: masood2004