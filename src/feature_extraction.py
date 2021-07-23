import pandas as pd
import re
import textstat

def sentence_statistics(df_text):
    """[summary]

    Args:
        df_sentence ([dataframe]): Pandas dataframe column with text paragraphs

    Returns:
        [dictionary]: python dictionary containing all computed statistics
    """

    # init dataframe
    df_stats = pd.DataFrame()

    # count comma
    df_stats["num_comma"] = df_text.str.count(",")

    # count number of sentences
    df_stats["num_sentences"] = df_text.apply(textstat.sentence_count)

    # count words
    df_stats["num_words"] = df_text.apply(textstat.lexicon_count)

    # count syllables
    df_stats["num_syllables"] = df_text.apply(textstat.syllable_count)

    # Flesch-reading ease 
    df_stats["flesch_reading_ease"] = df_text.apply(textstat.flesch_reading_ease)

    # Flesch-Kincaid Grade Level Formula 
    df_stats["flesch_grade_level"] = df_text.apply(textstat.flesch_kincaid_grade)

    # Gunning fog index
    df_stats["gunning_fog"] = df_text.apply(textstat.gunning_fog)

    # SMOG Index
    df_stats["SMOG_index"] = df_text.apply(textstat.smog_index)

    # Automated Readability Index
    df_stats["ARI"] = df_text.apply(textstat.automated_readability_index)

    # Coleman-Liau Index
    df_stats["coleman_liau"] = df_text.apply(textstat.coleman_liau_index)

    # Linsear Write Formular
    df_stats["linsear_write"] = df_text.apply(textstat.linsear_write_formula)

    # Dale Chall Readability
    df_stats["dale_chall"] = df_text.apply(textstat.dale_chall_readability_score)

    # Readability Consensus
    df_stats["combined_score"] = df_text.apply(textstat.text_standard, args=(True,))

    # preprocessing
    df_text = preprocessing(df_text)

    # count letters
    df_stats["num_letters"] = df_text.str.count(r"\w")

    return df_stats



def preprocessing(df_text):
    """Perform basic preprocessing such as lower casing, removing numbers, punctuations and multiple whitespaces. 

    Args:
        df_text ([dataframe]): Pandas dataframe column with text paragraphs

    Returns:
        [dataframe]: Pandas dataframe column with preprocessed text paragraphs
    """

    # lower case
    df_text = df_text.str.lower()

    # remove numbers
    df_text = df_text.apply(lambda x: re.sub(r"\d", "", x))

    # remove punctuations
    df_text = df_text.apply(lambda x: re.sub(r"\-", " ", x))
    df_text = df_text.apply(lambda x: re.sub(r"[^\w\s]", "", x))

    # reduce multiple whitespace to one whitespace
    df_text = df_text.apply(lambda x: re.sub(r"\s+", " ", x))

    return df_text


if __name__ == "__main__":
    test_sentence = pd.DataFrame(data=[["This is    test-sentence number 1 with a comma ,."],
                                  ["This is test-sentence    number ?!?! 2 with more numbers 21353215."],
                                  ["homomorphism"], ["a"]],
                                  columns=["sentences"])

    print(sentence_statistics(test_sentence.sentences))


