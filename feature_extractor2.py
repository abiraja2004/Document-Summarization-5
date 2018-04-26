import os
from math import log

import nltk

import preprocessing

current_directory = os.path.dirname(os.path.realpath(__file__))



def title_similarity(title,sentence, lemmatiser, stopwords_list):
	title=title.lower().split()
	filtered_title = [lemmatiser.lemmatize(preprocessing.remove_all_punctuation(word.strip()))
		      for word in title if word.strip() not in stopwords_list]
	sentence = [lemmatiser.lemmatize(preprocessing.remove_all_punctuation(word.strip()))
		      for word in sentence.lower().split() if word.strip() not in stopwords_list]
	common_words=[word for word in sentence if word in filtered_title]
	similarity=(len(common_words) * 0.1) / len(title)
	return similarity

def cue_words(sentence,lemmatiser):
	with open('cue_phrases') as f:
		cue_words_list = f.read().splitlines()
	cue_words_list=cue_words_list[:-1]
	cue_words_lemmatized=[lemmatiser.lemmatize(preprocessing.remove_all_punctuation(word.strip()))for word in cue_words_list]
	#print "cue_words_list",cue_words_list
	#print "cue_words_lemmatised",cue_words_lemmatized
	sentence=sentence.lower().split()
	cuewords_sentence=[]
	for word in sentence:
		if(word in cue_words_lemmatized):
			cuewords_sentence.append(word)
	increased_score=len(cuewords_sentence)*0.1
	return increased_score

def quotes_weighing(sentences):
	weights=[]
	check=[]
	for sentence in sentences:
		if(sentence.find("\"")!=-1):
			weights.append(0.2)
			check.append("yes")
		else:
			weights.append(1)	
			check.append("no")
	print check
	return weights

def ranking(document, tfidf_matrix, features, title, lemmatiser, stopwords_list):
	# print(document)
	sentences = nltk.sent_tokenize(document)
	print(sentences)
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
	#print(tokenized_sentences)
	tfidf_sentence = [[tfidf_matrix[features.index(word.lower())] for word in sentence if word.lower() in features]
		      for sentence in tokenized_sentences]
	# print(features)
	# print(tfidf_sentence)
	doc_val = sum(tfidf_matrix)
	sentence_score = [sum(sentence) / doc_val for sentence in tfidf_sentence]
	# print(sentence_score)
	similarity_with_title = [title_similarity(title, sentence, lemmatiser, stopwords_list) for sentence in sentences]
	print("sentence_score length", len(sentence_score))
	sentence_score = [sum(x) for x in zip(sentence_score, similarity_with_title)]

	print("title similarity", len(similarity_with_title))
	print("sentence_score length", len(sentence_score))
	# position weights
	mid = len(sentences) / 2
	print("number of sentences", len(tokenized_sentences))
	position_based_weighing = [sentence + (float(log((abs(mid - i) + 3))) / len(sentence_score))
		              if i > 0 else sentence + (float((abs(mid - i))) / len(sentence_score))
		              for i, sentence in enumerate(sentence_score)]
	# print(ranked_sentences)
	cue_weights=[cue_words(sentence,lemmatiser) for sentence in sentences]
	cue_based_weighing=[sum(x) for x in zip(position_based_weighing,cue_weights)]
	weights=quotes_weighing(sentences)
	final_weights=[a*b for a,b in zip(weights,cue_based_weighing)]
	ranked_sentences = [pair for pair in zip(range(len(sentence_score)), final_weights)]
	# print(ranked_sentences)
	ranked_sentences = sorted(ranked_sentences, key=lambda x: x[1] * -1)
	return ranked_sentences

