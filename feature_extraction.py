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
    df_stats["num_comma"] = df_sentence.str.count(",")

    # preprocessing
    df_sentence = preprocessing(df_sentence)

    # count words
    df_stats["num_words"] = df_sentence.str.split().str.len()


def preprocessing(df_sentence):

    # lower case
    df_sentence = df_sentence.str.lower()

    # remove numbers
    df_sentence = df_sentence.apply(lambda x: re.sub(r"\d", "", x))

    # remove punctuations
    df_sentence = df_sentence.apply(lambda x: re.sub(r"\-", " ", x))
    df_sentence = df_sentence.apply(lambda x: re.sub(r"[^\w\s]", "", x))

    # reduce multiple whitespace to one whitespace
    df_sentence = df_sentence.apply(lambda x: re.sub(r"\s+", " ", x))

    return df_sentence


if __name__ == "__main__":
    test_sentence = pd.DataFrame(data=[["This is    test-sentence number 1 with a comma ,."],
                                  ["This is test-sentence    number ?!?! 2 with more numbers 21353215."]],
                                  columns=["sentences"])

    print(preprocessing(test_sentence.sentences))


