import pandas as pd
import re

def sentence_statistics(df_sentence):
    """[summary]

    Args:
        df_sentence ([dataframe]): Pandas dataframe column with sentences
    """

    # init dataframe
    df_stats = pd.DataFrame()

    # count comma
    df_stats["num_comma"] = sentence.str.count(",")


def preprocessing(df_sentence):

    # lower case
    sentence = sentence.str.lower()

    # remove numbers

