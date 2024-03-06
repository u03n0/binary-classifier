import nltk
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from visualization import plot_label_counts, plot_num_of_stats


def run_eda_on_data(df: pd.DataFrame):
    new = df.copy()
    return add_stats(new)

def visualize_stats(df: pd.DataFrame):
    plot_label_counts(df)
    columns = ['num_of_chars', 'num_of_words', 'num_of_sents']
    for column_name in columns:
        plot_num_of_stats(df, column_name)

def add_stats(df: pd.DataFrame)-> pd.DataFrame:
    df['num_of_chars'] = df['text'].apply(len)
    df['num_of_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
    df['num_of_sents'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
    return visualize_stats(df)

