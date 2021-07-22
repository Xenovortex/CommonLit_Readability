import pandas as pd
import re

def sentence_statistics(df_sentence):
    """[summary]

    Args:
        df_sentence ([dataframe]): Pandas dataframe column with sentences

    Returns:
        [dictionary]: python dictionary containing all computed statistics
    """

    # init dataframe
    df_stats = pd.DataFrame()

    # count comma
    df_stats["num_comma"] = df_sentence.str.count(",")

    # preprocessing
    df_sentence = preprocessing(df_sentence)

    # count words
    df_stats["num_words"] = df_sentence.str.split().str.len()

    # count letters
    df_stats["num_letters"] = df_sentence.str.count(r"\w")

    # count syllables
    df_stats["num_syllables"] = df_sentence.apply(count_syllables)

    # count monosyllabic words 
    df_stats["num_monosyllables"] = df_sentence.apply(count_monosyllables)

    return df_stats



def count_syllables(sentence):
    """Count the number of syllables in a sentence

    Args:
        sentence ([string]): string that represent a sentence

    Returns: 
        [int]: number of syllables in the input sentence
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    sentence_syllables = 0
    words = sentence.split()
    for word in words:
        word_syllables = 1
        current_pos = len(word) - 1
        while current_pos >= 0:
            current_character = word[current_pos]
            current_pos -= 1
            if current_character in "aeiouyäöü":
                if current_pos <= 0:
                    break
                else:
                    current_character = word[current_pos]
                    if current_character not in "aeiouyäöü":
                        word_syllables += 1
                    current_pos -= 1
        if cc_pattern.match(word) and len(word) > 2:
            word_syllables -= 1
        sentence_syllables += word_syllables
    return sentence_syllables


def count_monosyllables(sentence):
    """Count monosyllabic words in a sentence

    Args:
        sentence -- string that represent a sentence

    Returns:
        [int]: number of monosyllabic words in the input sentence
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    monosyllables = 0
    words = sentence.split()
    for word in words:
        word_syllables = 1
        current_pos = len(word) - 1
        while current_pos >= 0:
            current_character = word[current_pos]
            current_pos -= 1
            if current_character in "aeiouyäöü":
                if current_pos <= 0:
                    break
                else:
                    current_character = word[current_pos]
                    if current_character not in "aeiouyäöü":
                        word_syllables += 1
                    current_pos -= 1
        if cc_pattern.match(word) and len(word) > 2:
            word_syllables -= 1
        if word_syllables == 1:
            monosyllables += 1
    return monosyllables


def preprocessing(df_sentence):
    """Perform basic preprocessing such as lower casing, removing numbers, punctuations and multiple whitespaces. 

    Args:
        df_sentence ([dataframe]): Pandas dataframe column with sentences

    Returns:
        [dataframe]: Pandas dataframe column with preprocessed sentences
    """

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

    print(sentence_statistics(test_sentence.sentences))


