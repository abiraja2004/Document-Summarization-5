import os

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import preprocessing
import util

current_directory = os.path.dirname(os.path.realpath(__file__))


# def title_similarity(title, sentence):
#     # Here we need to remove stopwords
#     title = title.lower().split()
#     sentence = sentence.lower.split()
#     common_words = [word for word in sentence if word in title]
#     similarity = (len(common_words) * 0.1) / len(t_tokens)  # Here need to think about 0.1
#     return similarity


def ranking(document, tfidf_matrix, features, summary_length):
    # print(document)
    sentences = nltk.sent_tokenize(document)
    # print(len(sentences))
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    # print(tokenized_sentences)
    # for i in range(len(sentences)):
    #	print("\n",sentences[i])
    # print(" ")
    tfidf_sentence = [[tfidf_matrix[features.index(word.lower())] for word in sentence if word.lower() in features]
                      for sentence in tokenized_sentences]
    # print(features)
    # print(tfidf_sentence)
    doc_val = sum(tfidf_matrix)
    sentence_score = [sum(sentence) / doc_val for sentence in tfidf_sentence]
    # print(sentence_score)
    # similarity_with_title=[title_similarity(title, sentence) for sentence in sentences]
    # position weights
    mid = len(sentence_score) / 2
    print("number of sentences", len(tokenized_sentences))
    ranked_sentences = [sentence * (abs(mid - i) / len(sentence_score)) for i, sentence in enumerate(sentence_score)]
    # print(ranked_sentences)
    ranked_sentences = [pair for pair in zip(range(len(sentence_score)), sentence_score)]
    # print(ranked_sentences)
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[1] * -1)
    return ranked_sentences[:summary_length]