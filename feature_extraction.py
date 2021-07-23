import pandas as pd
import re

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
    df_stats["num_sentences"] = df_text.str.count(".") + df_text.str.count("?") + df_text.str.count("!")

    # preprocessing
    df_text = preprocessing(df_text)

    # count words
    df_stats["num_words"] = df_text.str.split().str.len()

    # count letters
    df_stats["num_letters"] = df_text.str.count(r"\w")

    # count syllables
    df_stats["num_syllables"] = df_text.apply(count_syllables)

    # count monosyllabic words 
    df_stats["num_monosyllables"] = df_text.apply(count_monosyllables)

    # count polysyllabic words
    df_stats["num_polysyllables_2"] = df_text.apply(count_polysyllables, args=(2,))
    df_stats["num_polysyllables_3"] = df_text.apply(count_polysyllables, args=(3,))
    df_stats["num_polysyllables_5"] = df_text.apply(count_polysyllables, args=(5,))

    # count long words
    df_stats["num_long_3"] = df_text.apply(count_long_words, args=(3,))
    df_stats["num_long_5"] = df_text.apply(count_long_words, args=(5,))
    df_stats["num_long_8"] = df_text.apply(count_long_words, args=(8,))
    df_stats["num_long_10"] = df_text.apply(count_long_words, args=(10,))
    df_stats["num_long_15"] = df_text.apply(count_long_words, args=(15,))

    # Flesch-reading ease (https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
    df_stats["flesch_reading_ease"] = 206.835 - 1.015 * (df_stats["num_words"] / df_stats["num_sentences"]) - 84.6 * (df_stats["num_syllables"] / df_stats["num_words"])

    # Flesch-Kincaid Grade Level Formula (https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
    df_stats["flesch_grade_level"] = 0.39 * (df_stats["num_words"] / df_stats["num_sentences"]) + 11.8 * (df_stats["num_syllables"] / df_stats["num_words"]) - 15.59

    # Flesch formular Farr, Jenkins and Patterson modification (https://en.wikipedia.org/wiki/Readability)
    df_stats["flesch_modified"] = 1.599 * (df_stats["num_monosyllables"] * 100 / df_stats["num_words"]) - 1.015 * (df_stats["num_words"] / df_stats["num_sentences"]) - 31.517

    # 

    return df_stats



def count_syllables(text):
    """Count the number of syllables in a text paragraph

    Args:
        text ([string]): string that represent a text paragraph

    Returns: 
        [int]: number of syllables in the input text paragraph
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    sentence_syllables = 0
    words = text.split()
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


def count_monosyllables(text):
    """Count monosyllabic words in a text paragraph

    Args:
        text ([string]): string that represent a text paragraph

    Returns:
        [int]: number of monosyllabic words in the input text paragraph
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    monosyllables = 0
    words = text.split()
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


def count_polysyllables(text, threshold=2):
    """Count polysyllabic words that have equal or more syllables than the threshold

    Args:
        text ([string]): string that represent a text paragraph
        threshold ([int], optional): syllable threshold for polysyllabic words to consider
    
    Returns:
        [int]: number of polysyllabic words in the input text paragraph with more syllables than the threshold
    """
    cc_pattern = re.compile("[^aeiouyäöü]{2,}")
    polysyllables = 0
    words = text.split()
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
        if word_syllables >= threshold:
            polysyllables += 1
    return polysyllables


def count_long_words(text, threshold):
    """Count words in a text paragraph that have equal or more letters than the threshold

    Args:
        text ([string]): string that represent a text paragraph
        threshold ([int]): threshold for number of letters in a word for it to classify as a long word

    Returns:
        [int]: number of words in the input text paragraph that have more letters than the threshold
    """
    long_words = 0
    words = text.split()
    for word in words:
        if len(word) >= threshold:
            long_words += 1
    return long_words


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
                                  ["This is test-sentence    number ?!?! 2 with more numbers 21353215."]],
                                  columns=["sentences"])

    print(sentence_statistics(test_sentence.sentences))


