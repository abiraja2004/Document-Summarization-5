import util
import feature_extractor
import preprocessing
import os
import config

current_directory = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':

    clean_data = preprocessing.pre_process()

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

    top_sentences = feature_extractor.ranking(clean_data[7]['article'], doc_matrix, features, config.SUMMARY_LENGTH)

    print(clean_data[7]['article'])
    summary = '.'.join([clean_data[7]['article'].split('.')[i]
                        for i in [pair[0] for pair in top_sentences]])
    summary = ' '.join(summary.split())
    print("summary")
    print(summary)