import pandas as pd
from nltk.corpus import stopwords
import string


def main_normalization(df):
    df = df.drop_duplicates()

    df = df.fillna(0)

    df['AdText'] = df['AdText'].apply(clean_text)
    df['Clicks'] = df['Clicks'].apply(clean_numbers, args=(40000,))
    df['Clicks'] = df['Clicks'].apply(lambda x: 1 if x < 1 and pd.notna(x) else x)
    df['Clicks'] = df['Clicks'].astype('Int64', errors='ignore')

    df['Impressions'] = df['Impressions'].apply(clean_numbers, args=(400000,))
    df['Impressions'] = df['Impressions'].apply(lambda x: 1 if x < 1 and pd.notna(x) else x)
    df['Impressions'] = df['Impressions'].astype('Int64', errors='ignore')
    return df


def clean_text(text):
    if isinstance(text, str):  # if text is string
        stop_words = set(stopwords.words('english'))
        translator = str.maketrans('', '', string.punctuation)

        # Deletion of punctuation and splitting text by words
        words = text.translate(translator).split()

        # Deletion of stop-words
        words = [word.lower() for word in words if word.lower() not in stop_words]

        return ' '.join(words)
    else:
        return ''


def clean_numbers(text, max_value):
    if text > max_value:
        return None
    else:
        return text
