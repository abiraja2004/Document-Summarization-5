import os
from math import log

import nltk

import config
import preprocessing

current_directory = os.path.dirname(os.path.realpath(__file__))


def title_similarity(title,sentence, lemmatiser, stopwords_list):
    title=title.lower().split()
    filtered_title = [lemmatiser.lemmatize(preprocessing.remove_all_punctuation(word.strip()))
                      for word in title if word.strip() not in stopwords_list]
    sentence = [lemmatiser.lemmatize(preprocessing.remove_all_punctuation(word.strip()))
                      for word in sentence.lower().split() if word.strip() not in stopwords_list]
    common_words=[word for word in sentence if word in filtered_title]
    similarity=(len(common_words) * config.TITLE_SIMILARITY_WEIGHT) / len(title)

    return similarity


def cue_words(sentence,lemmatiser):
    cue_words_list=['summary', 'particular', 'important', 'conclusion', 'infer', 'report']
    cue_words_lemmatized=[lemmatiser.lemmatize(preprocessing.remove_all_punctuation(word.strip()))for word in cue_words_list]
    sentence=sentence.lower().split()
    cuewords_sentence=[]
    for word in sentence:
        if(word in cue_words_lemmatized):
            cuewords_sentence.append(word)
    increased_score=len(cuewords_sentence)*config.CUE_WORDS_WEIGHT
    return increased_score


def ranking(document, tfidf_matrix, features, title, lemmatiser, stopwords_list):
    sentences = nltk.sent_tokenize(document)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    tfidf_sentence = [[tfidf_matrix[features.index(word.lower())] for word in sentence if word.lower() in features]
                      for sentence in tokenized_sentences]

    doc_val = sum(tfidf_matrix)
    sentence_score = [sum(sentence) / doc_val for sentence in tfidf_sentence]

    similarity_with_title = [title_similarity(title, sentence, lemmatiser, stopwords_list) for sentence in sentences]
    cue_weights = [cue_words(sentence, lemmatiser) for sentence in sentences]
    sentence_score = [sum(x) for x in zip(sentence_score, similarity_with_title,cue_weights)]

    # position weights
    mid = len(sentences) / 2
    positon_based_weighing = [sentence + (float(log((abs(mid - i) + 3))) / len(sentence_score))
                              if i > 0 else sentence + (float((abs(mid - i))) / len(sentence_score))
                              for i, sentence in enumerate(sentence_score)]

    ranked_sentences = [pair for pair in zip(range(len(sentence_score)), positon_based_weighing)]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[1] * -1)

    return ranked_sentences