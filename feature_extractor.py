import os
from math import log

import nltk

current_directory = os.path.dirname(os.path.realpath(__file__))


def title_similarity(title,sentence):
    #Here we need to remove stopwords
    title=title.lower().split()
    sentence=sentence.lower().split()
    common_words=[word for word in sentence if word in title]
    similarity=(len(common_words) * 0.1 ) / len(title) #Here need to think about 0.1
    return similarity


def ranking(document, tfidf_matrix, features, summary_length, title):
    # print(document)
    sentences = nltk.sent_tokenize(document)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    # print(tokenized_sentences)
    tfidf_sentence = [[tfidf_matrix[features.index(word.lower())] for word in sentence if word.lower() in features]
                      for sentence in tokenized_sentences]
    # print(features)
    # print(tfidf_sentence)
    doc_val = sum(tfidf_matrix)
    sentence_score = [sum(sentence) / doc_val for sentence in tfidf_sentence]
    # print(sentence_score)
    similarity_with_title = [title_similarity(title, sentence) for sentence in sentences]
    print("sentence_score length", len(sentence_score))
    sentence_score = [sum(x) for x in zip(sentence_score, similarity_with_title)]

    print("title similarity", len(similarity_with_title))
    print("sentence_score length", len(sentence_score))
    # position weights
    mid = len(sentences) / 2
    print("number of sentences", len(tokenized_sentences))
    positon_based_weighing = [sentence + (log((abs(mid - i) + 3)) / len(sentence_score)) for i, sentence in enumerate(sentence_score)]
    # print(ranked_sentences)
    ranked_sentences = [pair for pair in zip(range(len(sentence_score)), positon_based_weighing)]
    # print(ranked_sentences)
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[1] * -1)
    return ranked_sentences[:summary_length]