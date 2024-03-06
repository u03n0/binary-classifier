import nltk
import pandas as pd
from typing import Callable
from visualization import plot_label_counts, plot_num_of_stats


def run_eda_on_data(df: pd.DataFrame)-> Callable:
    """Begins process of performing EDA on pre-processed data
    """
    new = df.copy()
    return add_stats(new)

def visualize_stats(df: pd.DataFrame):
    """ creates and saves plots for stats.
    """
    plot_label_counts(df)
    columns = ['num_of_chars', 'num_of_words', 'num_of_sents']
    for column_name in columns:
        plot_num_of_stats(df, column_name)

def add_stats(df: pd.DataFrame)-> Callable:
    """ Adds stats to df.
    """
    df['num_of_chars'] = df['text'].apply(len)
    df['num_of_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
    df['num_of_sents'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
    return visualize_stats(df)

