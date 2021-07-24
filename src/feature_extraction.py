import pandas as pd
import re
import nltk
from collections import Counter
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

    # reduce multiple whitespace to one whitespace
    df_text = df_text.apply(lambda x: re.sub(r"\s+", " ", x))

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

    # lower case
    df_text = df_text.str.lower()

    # Count POS tags
    df_tokens = df_text.apply(nltk.word_tokenize)
    df_pos = df_tokens.apply(nltk.pos_tag)
    df_pos_stats = count_POS_tag(df_pos)
    df_stats = pd.concat([df_stats, df_pos_stats], axis=1)


    return df_stats


def count_POS_tag(df_pos):
    """Count how often each POS tag occurs

    Args:
        df_pos ([dataframe]): dataframe, where the entries are list of tuples (token, POS tag) 

    Returns:
        df_pos_stats ([dataframe]): dataframe containing POS tag statistics
    """

    # POS tag list
    tag_lst = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 
               'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
               'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '$', "''", '(', ')', ',', '.', ':', '``'] 
    
    # init dataframe
    df_pos_stats = pd.DataFrame(0, index=range(len(df_pos)), columns=tag_lst)

    # count POS tag
    for index, pos in enumerate(df_pos):
        count_dict = Counter(tag for _, tag in pos)
        for tag, count in count_dict.items():
            if tag in tag_lst: 
                df_pos_stats.loc[index, tag] = count

    return df_pos_stats


if __name__ == "__main__":
    test_sentence = pd.DataFrame(data=[["This is    test-sentence number 1 with a comma ,."],
                                  ["This is test-sentence    number ?!?! 2 with more numbers 21353215."],
                                  ["homomorphism"], ["--"], ['"""_-()[],--?!:$;...``']],
                                  columns=["sentences"])

    print(sentence_statistics(test_sentence.sentences))

    #print(nltk.help.upenn_tagset())