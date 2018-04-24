import os
import argparse
from collections import Counter
from math import sqrt

import nltk
from nltk.stem import WordNetLemmatizer

import config
import feature_extractor
import preprocessing
import util

current_directory = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-a', dest='article_no', help='Article No which needs to be summarised')
    arguments = argparser.parse_args()
    return arguments

def get_stopwords():
    stopwords = open(os.path.join(current_directory,'stopwords.txt'))
    stopwords_list = [stopword.strip() for stopword in stopwords]
    stopwords.close()
    stopwords_list_lemmatised = [lemmatiser.lemmatize(stopword) for stopword in stopwords_list]
    return stopwords_list_lemmatised

def cosine_similarity(sent1, sent2, lemmatiser, stopwords_list):

    filtered_sent1 = [lemmatiser.lemmatize(preprocessing.remove_all_punctuation(word.strip()))
                      for word in sent1.split(' ') if word.strip() not in stopwords_list]
    filtered_sent2 = [lemmatiser.lemmatize(preprocessing.remove_all_punctuation(word.strip()))
                      for word in sent2.split(' ') if word.strip() not in stopwords_list]

    freq_sent1 = Counter(filtered_sent1)
    freq_sent2 = Counter(filtered_sent2)

    common_words = set(freq_sent1.keys()) & set(freq_sent2.keys())

    numerator = sum([freq_sent1[word] * freq_sent2[word] for word in common_words])

    sum1 = sum([freq_sent1[x] ** 2 for x in freq_sent1.keys()])
    sum2 = sum([freq_sent2[x] ** 2 for x in freq_sent2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def generate_summary_cosine_similarity(tokenized_sentences, ranked_sentences, summary_length, lemmatiser, stopwords_list):

    def similarity_calculate(summary_sentences, sentence):
        no_of_similar_sentences = 0
        for sentence_details in summary_sentences.values():
            if cosine_similarity(tokenized_sentences[sentence_details[0][0]],
                                 tokenized_sentences[sentence], lemmatiser, stopwords_list) > config.SIMILARITY_THRESHOLD:
                no_of_similar_sentences+=1

        return no_of_similar_sentences

    sentences_similarity = {0 : [ranked_sentences[0]]}
    # print summary_sentences
    for i in xrange(1,len(ranked_sentences)):
        similarity = similarity_calculate(sentences_similarity,ranked_sentences[i][0])
        if similarity not in sentences_similarity:
            sentences_similarity[similarity] = []
        sentences_similarity[similarity].append(ranked_sentences[i])

    # print sentences_similarity
    if len(sentences_similarity[0]) >= summary_length:
        top_sentences = sorted(sentences_similarity[0], key=lambda x: x[1] * -1)[:summary_length]
    else:
        top_sentences = []
        sentences_needed = summary_length - len(sentences_similarity[0])
        for similarity_score in sorted(sentences_similarity).keys():
            # if sentences_needed <= 0:
            #     break
            for sentence in sorted(sentences_similarity[similarity_score], key=lambda x: x[1] * -1):
                if sentences_needed <= 0:
                    break
                top_sentences.append(sentence)
                sentences_needed-=1
    return top_sentences


def generate_summary(tokenized_sentences, ranked_sentences, lemmatiser, stopwords_list):

    if config.SUMMARY_LENGTH < 1:
        summary_length = len(tokenized_sentences)*config.SUMMARY_LENGTH
    else:
        summary_length = min(config.SUMMARY_LENGTH, len(tokenized_sentences))

    if config.USE_SIMILARITY:
        top_sentences = generate_summary_cosine_similarity(tokenized_sentences, ranked_sentences, summary_length,
                                                           lemmatiser, stopwords_list)
    else:
        top_sentences = ranked_sentences[:summary_length]

    summary = '.'.join([ tokenized_sentences[i]
                        for i in [pair[0] for pair in top_sentences]])
    summary = ' '.join(summary.split())
    return summary

if __name__ == '__main__':


    article_no = int(parse_args().article_no)
    print article_no
    clean_data = preprocessing.pre_process()

    lemmatiser = WordNetLemmatizer()
    stopwords_list = get_stopwords()

    # corpus_data = [record['article'] for record in clean_data]
    corpus_data = map(lambda record: record['article'], clean_data)
    corpus_data = set(corpus_data)
    print "Size of Corpus Data: {}".format(len(corpus_data))

    # count_vect = CountVectorizer()
    # count_vect = count_vect.fit(corpus_data)
    # util.save_to_disk(count_vect, current_directory + '/pickle_objects/count_vect')
    count_vect = util.load_from_disk(current_directory + '/pickle_objects/count_vect')
    freq_term_matrix = count_vect.transform(corpus_data)
    features = count_vect.get_feature_names()

    # tfidf = TfidfTransformer(norm="l2")
    # tfidf.fit(freq_term_matrix)
    # util.save_to_disk(tfidf, current_directory + '/pickle_objects/tf_idf')
    tfidf = util.load_from_disk(current_directory + '/pickle_objects/tf_idf')
    trans_freq_term_matrix = count_vect.transform(record['article'] for record in clean_data)
    transformed_tfidf_matrix = tfidf.transform(trans_freq_term_matrix)

    transformed_dense = transformed_tfidf_matrix.todense()
    print(len(transformed_dense))
    doc_matrix = transformed_dense[0:40].tolist()[0]  # Here we are supposed to put the entire transformed_dense

    list2 = []
    tokenized_sentences = nltk.sent_tokenize(clean_data[article_no]['article'])

    ranked_sentences = feature_extractor.ranking(clean_data[article_no]['article'], doc_matrix, features,
                                                 clean_data[article_no]['title'], lemmatiser, stopwords_list)
    print clean_data[article_no]['title']
    print(clean_data[article_no]['article'])

    summary = generate_summary(tokenized_sentences,ranked_sentences, lemmatiser, stopwords_list)
    # article_sentences = nltk.sent_tokenize(clean_data[8]['article'])
    # summary = '.'.join([ article_sentences[i]
    #                     for i in [pair[0] for pair in top_sentences]])
    # summary = ' '.join(summary.split())
    print("summary")
    print(summary)