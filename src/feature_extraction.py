import pandas as pd
import numpy as np
import re
import nltk
from collections import Counter
import textstat
from simplerepresentations import RepresentationModel


def generate_representation(df_text, model_type, model_name, batch_size, max_length, combine_method, num_hidden, save_path=None):
    """Generate feature representation using the specified pretrained model

    Args:
        df_text ([dataframe]): Pandas dataframe column with text paragraphs
        model_type ([string]): model type (see HuggingFace Transformers library)
        model_name ([string]): model name (see HuggingFace Transformers library)
        batch_size ([int]): size of batch
        max_length ([int]): if more tokens than max_length, truncate such that there are less than or equal max_length tokens
        combine_method ([string]): method to combine hidden states. Options: 'cat' or 'sum'
        num_hidden ([int]): number of last hiddent states used to generate representation
        save_path ([string], optional): path to save the representation as hdf5 file. Defaults to None.

    Returns:
        [type]: [description]
    """

    model = RepresentationModel(
        model_type = model_type,
        model_name = model_name,
        batch_size = batch_size,
        max_seq_length = max_length,
        combination_method = combine_method,
        last_hidden_to_use = num_hidden
    )

    sentence_features, token_features = model(df_text.values)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            np.save(f, sentence_features)
            np.save(f, token_features)

    return sentence_features, token_features



def generate_statistics(df_text):
    """Generate statistics for a given paragraph text 

    Args:
        df_text ([dataframe]): Pandas dataframe column with text paragraphs

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

    # count characters
    df_stats["num_char"] = df_text.apply(len)

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
    df_stats["combined_score"] = df_text.apply(textstat.text_standard)

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
    test_sentence = pd.DataFrame(data=[["This is test-sentence number 1 with a comma ,."],
                                  ["This is test-sentence number 2 with more numbers 21353215."],
                                  ["homomorphism"]],
                                  columns=["sentences"])

    print(generate_representation(test_sentence.sentences, 'roberta', 'roberta-base', 128, 128, 'sum', 4))

    #print(nltk.help.upenn_tagset())