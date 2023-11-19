from gensim import corpora, models
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


# Applying LDA for each cluster
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return tokens


def make_lda(input_file_name, output_file_name):
    df = pd.read_csv(input_file_name)
    Y = df[['Clicks', 'Impressions', 'AdText', 'cluster']].dropna().copy()

    # Iterate over each cluster
    for cluster_id in range(5):
        # Filter data for the current cluster
        cluster_data = Y[Y['cluster'] == cluster_id]['AdText'].values.tolist()

        # Tokenize and preprocess text
        tokenized_text = [preprocess_text(text) for text in cluster_data]

        # Create a dictionary
        dictionary = corpora.Dictionary(tokenized_text)

        # Create a corpus
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_text]

        # Train the LDA model
        lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

        # Display topics for the current cluster
        print(f"\nCluster {cluster_id} Topics:")
        for topic, words in lda_model.print_topics():
            print(f"Topic {topic}: {words}")

        # Apply LDA to each document in the current cluster
        lda_topics = [lda_model.get_document_topics(dictionary.doc2bow(preprocess_text(x))) for x in cluster_data]

        # Extract only the topic probabilities from the LDA result
        lda_topic_values = [dict(topic) for topic in lda_topics]

        # Add LDA topics to the DataFrame
        Y.loc[Y['cluster'] == cluster_id, 'lda_topic'] = lda_topic_values

    num_topics = lda_model.num_topics

    # Matrix of zeros sized by (number of documents) x (number of topics)
    matrix = np.zeros((len(lda_topics), num_topics))

    # Fill in matrix
    for i, doc in enumerate(lda_topics):
        for topic, weight in doc:
            matrix[i, topic] = weight

    # Create DataFrame with matrix
    df_matrix = pd.DataFrame(matrix, columns=[f'Topic {i}' for i in range(num_topics)])

    # Create temperature map
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_matrix, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Topic Weight'})
    plt.title('Topic Distribution Across Documents')
    plt.xlabel('Topics')
    plt.ylabel('Documents')
    plt.show()

    # Save the DataFrame with LDA topics
    df_with_lda = pd.concat([df, Y['lda_topic']], axis=1)
    df_with_lda.to_csv(output_file_name, index=False)
