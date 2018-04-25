import os
import re
import string

import util

# from io import open
current_directory = os.path.dirname(os.path.realpath(__file__))


def news_finder(data):
    news_data = [{'title':record['headlines'],'article':record['ctext']} for record in data]
    return news_data

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', ' ', text)

def remove_consecutive_spaces(text):
    return re.sub('\s+',' ', text)

def remove_all_punctuation(text):
    return text.translate(None, string.punctuation).strip()

def clean_text(text):
    raw_text = remove_non_ascii(text)
    raw_text = re.sub('\s+', ' ', raw_text).strip()
    raw_text = re.sub('([.])(\S\S)',r'. \2', raw_text)
    raw_text = re.sub('([.])([A])', r'. \2', raw_text)
    # re_list = [['\.*\s*\?', '.\"'], ['\?*\s*\.', '\".'],['\,*\s*\?', ',\"'],
    #            ['\?*\s*\,', '\",'],['\;*\s*\?', ';\"'],['\?*\s*\;', '\";']]
    # for expr in re_list:
    #     raw_text = re.sub(expr[0],expr[1],raw_text)
    return raw_text

def cleanse_data(data):
    clean_data = []
    for record in data:
        record['title'] = clean_text(remove_all_punctuation(record['title']))
        record['article'] = clean_text(record['article'])
        clean_data.append(record)
    return clean_data

# if __name__ == '__main__':
def pre_process():
    with open(current_directory + '/news_summary.csv', "rb") as corpus:
        csv_data = util.csv_dict_reader(corpus)
        news_data = news_finder(csv_data)
    clean_data = cleanse_data(news_data)
    return clean_data