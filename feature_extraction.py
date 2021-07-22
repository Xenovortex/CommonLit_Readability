import pandas as pd

def sentence_statistics(sentence):

    # init dataframe
    df_stats = pd.DataFrame()

    # count comma
    df_stats["num_comma"] = sentence.str.count(",")