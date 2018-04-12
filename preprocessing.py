import util
import os
current_directory = os.path.dirname(os.path.realpath(__file__))

def newsFinder(data):
    news_data = [{'title':record[2],'article':record[5]} for record in data]
    return news_data


if __name__ == '__main__':
    with open(current_directory + '/news_summary.csv', "rb") as corpus:
        csv_data = util.csv_reader(corpus)