from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return tokens


df = pd.read_csv("ads_with_clusters_and_lda.csv")

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
    # print(f"\nCluster {cluster_id} Topics:")
    # for topic, words in lda_model.print_topics():
    #     print(f"Topic {topic}: {words}")

    # Apply LDA to each document in the current cluster
    lda_topics = [lda_model.get_document_topics(dictionary.doc2bow(preprocess_text(x))) for x in cluster_data]

    # Extract only the topic probabilities from the LDA result
    lda_topic_values = [dict(topic) for topic in lda_topics]

    # Add LDA topics to the DataFrame
    Y.loc[Y['cluster'] == cluster_id, 'lda_topic'] = lda_topic_values

# Виведення та обробка кожної теми
for topic_id, topic_words in lda_topics:
    # Розбивка рядка тем на слова
    topic_words_list = [word.split("*")[1].strip().strip('"') for word in topic_words.split('+')]

# Отримайте текст тем та їх ваги
topics = lda_topics

# Об'єднайте слова з усіх тем
all_words = " ".join([str(word) for topic_words_list in topics for word in topic_words_list[1].split('+')])

# Створіть об'єкт WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)


# Виведіть хмару слів
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud.png')
# plt.show()
