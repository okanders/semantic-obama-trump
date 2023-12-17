import os
import re
import numpy as np
import pandas as pd

from wordcloud import WordCloud

import gensim
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import pipeline
from sentence_transformers import SentenceTransformer


from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def wordcloud(pre_trump_top_words,post_trump_top_words, top_words):
        # Create a dictionary of the top words associated with pre-Trump, post-Trump, and overall
    pre_trump_words_dict = dict(pre_trump_top_words)
    post_trump_words_dict = dict(post_trump_top_words)
    overall_words_dict = dict(top_words)

    # Create a word cloud for the top words associated with pre-Trump
    pre_trump_wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(pre_trump_words_dict)

    # Create a word cloud for the top words associated with post-Trump
    post_trump_wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(post_trump_words_dict)

    # Create a word cloud for the top words overall
    overall_wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(overall_words_dict)

    # Display the word clouds using Matplotlib
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(pre_trump_wordcloud, interpolation='bilinear')
    axs[0].set_title('Top words associated with pre-Trump')
    axs[0].axis('off')
    axs[1].imshow(post_trump_wordcloud, interpolation='bilinear')
    axs[1].set_title('Top words associated with post-Trump')
    axs[1].axis('off')
    axs[2].imshow(overall_wordcloud, interpolation='bilinear')
    axs[2].set_title('Top words overall')
    axs[2].axis('off')
    plt.show()

def logit_lasso(data, labels):
        # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}


    # Train the model
    model = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5)
    model.fit(X_train_vec, y_train)

    # Evaluate the model on the test set using the best hyperparameters
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")

    # Analyze the words
    coef = model.best_estimator_.coef_[0]
    words = vectorizer.get_feature_names_out()

    # Get the top 10 most positive and negative coefficients
    #In other words, the two processes are complementary, with the first process providing a 
    # general view of the words that are most strongly associated with sentiment across the entire dataset,
    #  and the second process providing a more nuanced view of the words that are most strongly associated 
    # with each period individually.
    sorted_coef_idx = coef.argsort()
    top_negative = [(words[i], coef[i]) for i in sorted_coef_idx[:10]]
    top_positive = [(words[i], coef[i]) for i in sorted_coef_idx[:-11:-1]]

    print(f"Top negative coefficients: {top_negative}")
    print(f"Top positive coefficients: {top_positive}")

    # Top words associated with pre-Trump
    pre_trump_top_words = [(word, coef[idx]) for idx, word in enumerate(words) if coef[idx] < 100]
    pre_trump_top_words = sorted(pre_trump_top_words, key=lambda x: x[1])[:10]

    # Top words associated with post-Trump
    post_trump_top_words = [(word, coef[idx]) for idx, word in enumerate(words) if coef[idx] > -100]
    post_trump_top_words = sorted(post_trump_top_words, key=lambda x: x[1], reverse=True)[:10]

    # Top words overall
    top_words = [(word, coef[idx]) for idx, word in enumerate(words)]
    top_words = sorted(top_words, key=lambda x: x[1], reverse=True)[:20]

    # Top words associated with positive sentiment
    pos_top_words = [(word, coef[idx]) for idx, word in enumerate(words) if coef[idx] > 0]
    pos_top_words = sorted(pos_top_words, key=lambda x: x[1], reverse=True)[:10]

    # Top words associated with negative sentiment
    neg_top_words = [(word, coef[idx]) for idx, word in enumerate(words) if coef[idx] < 0]
    neg_top_words = sorted(neg_top_words, key=lambda x: x[1])[:10]



    # Print the results
    print("Top words associated with pre-Trump:")
    for word, score in pre_trump_top_words:
        print(f"{word}: {score:.3f}")
    print()

    print("Top words associated with post-Trump:")
    for word, score in post_trump_top_words:
        print(f"{word}: {score:.3f}")
    print()

    print("Top words overall:")
    for word, score in top_words:
        print(f"{word}: {score:.3f}")
    print()

    print("Top words associated with positive sentiment:")
    for word, score in pos_top_words:
        print(f"{word}: {score:.3f}")
    print()

    print("Top words associated with negative sentiment:")
    for word, score in neg_top_words:
        print(f"{word}: {score:.3f}")
    print()


    return pre_trump_top_words, post_trump_top_words, X_train_vec, top_words

def preprocess_text(text):
    # Remove any occurrence of 'trump' or 'obama' from the text
    text = re.sub(r'\b(trump|obama)\b', '', text, flags=re.IGNORECASE)
    return text


def sentence_transform(obama_df, trump_df):
    # Load the pre-trained sentence-transformer model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Embed the Obama and Trump headlines using the sentence-transformer model
    obama_embeddings = model.encode(obama_df['headline'].tolist())
    trump_embeddings = model.encode(trump_df['headline'].tolist())

    # Compute cosine similarity between all pairs of Obama and Trump headline embeddings
    similarity_matrix = cosine_similarity(obama_embeddings, trump_embeddings)

    # Get the indices of the top 10 similarity scores
    top_indices = similarity_matrix.flatten().argsort()[-10:][::-1]

    # Print the top 10 similarity scores and the respective headlines
    for index in top_indices:
        i, j = divmod(index, similarity_matrix.shape[1])
        similarity_score = similarity_matrix[i, j]
        obama_headline = obama_df['headline'][i]
        trump_headline = trump_df['headline'][j]
        print(f"Similarity score: {similarity_score}")
        print(f"Obama headline: {obama_headline}")
        print(f"Trump headline: {trump_headline}\n")

def transform(df):

    # 0: "AGAINST", 1: "FAVOR", 2: "NONE"
    sentiment_pipeline = pipeline("sentiment-analysis", model="kornosk/bert-election2020-twitter-stance-trump")

    #Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
    #sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


    
    # Get the first 10 headlines from the combined DataFrame
    headlines = df[0:10].tolist()

    # Apply sentiment analysis pipeline to the first 10 headlines
    sentiments = sentiment_pipeline(headlines)

    # Print the sentiment predictions and original headlines
    for i, headline in enumerate(headlines):
        print(f"Headline {i+1}: {headline}")
        print(f"Sentiment: {sentiments[i]['label']}, Score: {sentiments[i]['score']}\n")

def word_count(obama_articles, trump_articles):

    # create a list of words for each dataset
    obama_words = [headline.split() for headline in obama_articles]
    trump_words = [headline.split() for headline in trump_articles]

    # count the frequency of each word in each dataset
    obama_word_counts = Counter(word for words in obama_words for word in words)
    trump_word_counts = Counter(word for words in trump_words for word in words)

    # get the top 10 words for each dataset
    obama_top_words = obama_word_counts.most_common(30)
    trump_top_words = trump_word_counts.most_common(30)

    # print out the top 10 words for each dataset
    print("Top 10 words in Obama articles:")
    for word, count in obama_top_words:
        print(f"{word}: {count}")

    print("\nTop 10 words in Trump articles:")
    for word, count in trump_top_words:
        print(f"{word}: {count}")

def cosine_sim(pre_trump_top_words,post_trump_top_words):

    # Create a list of all the top words
    top_words = pre_trump_top_words + post_trump_top_words

    # Create a Word2Vec model using the top words
    model = gensim.models.Word2Vec(top_words, min_count=1)

    # Calculate the cosine similarity between the top words
    pre_trump_similarities = []
    post_trump_similarities = []

    for word, _ in pre_trump_top_words:
        similarity = model.wv.similarity(word, 'trump')
        pre_trump_similarities.append((word, similarity))

    for word, _ in post_trump_top_words:
        similarity = model.wv.similarity(word, 'trump')
        post_trump_similarities.append((word, similarity))

    # Print the top words and their cosine similarities
    print('Top words associated with pre-Trump and their cosine similarities to "trump":', pre_trump_similarities)
    print('Top words associated with post-Trump and their cosine similarities to "trump":', post_trump_similarities)

def main():
    obama_df = pd.read_csv('obama.csv')
    trump_df = pd.read_csv('trump.csv')
    
    # Preprocess the headline text
    #obama_df['headline'] = obama_df['headline'].apply(preprocess_text)
    #trump_df['headline'] = trump_df['headline'].apply(preprocess_text)


    #sentence_transform(obama_df, trump_df)

    obama_articles = obama_df['headline']
    trump_articles = trump_df['headline']

    #word_count(obama_articles, trump_articles)

    # Combine and label data
    data = pd.concat([obama_articles, trump_articles])

    transform(obama_articles)
    transform(trump_articles)

    #hot encoding
    labels = [0] * len(obama_articles) + [1] * len(trump_articles)

    pre_trump_top_words, post_trump_top_words, X_train_vec, top_words = logit_lasso(data, labels)


    # Get the vectors for the top 10 pre-Trump words and top 10 post-Trump words
    #pre_trump_vecs = [X_train_vec[:, words.tolist().index(word)].toarray() for (word, score) in pre_trump_top_words]
    #post_trump_vecs = [X_train_vec[:, words.tolist().index(word)].toarray() for (word, score) in post_trump_top_words]
    #cosine_sim(pre_trump_top_words,post_trump_top_words)

    # Calculate the cosine similarity matrix between the pre-Trump and post-Trump word vectors
    similarity_matrix = cosine_similarity(X_train_vec)
    # Find the indices of the top 10 similarity scores
    top_indices = similarity_matrix.argsort()[-10:][::-1]

    # Print the top 10 similarity scores and the respective headlines
    for index in top_indices:
        i, j = divmod(index, similarity_matrix.shape[1])
        similarity_score = similarity_matrix[i, j]
        headline = X_train_vec[i]
        print(f"Similarity score: {similarity_score}")
        print(f"Headline: {headline}\n")
    
    # create a heatmap of the similarity matrix
   # sns.heatmap(similarity_matrix, cmap="YlGnBu")
   # plt.title("Cosine Similarity Matrix")
   # plt.xlabel("Article IDs")
   # plt.ylabel("Article IDs")
   # plt.show()

    wordcloud(pre_trump_top_words,post_trump_top_words, top_words)


if __name__ == '__main__':
    main()
