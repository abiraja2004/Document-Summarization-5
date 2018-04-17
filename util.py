import pickle
import csv

def csv_dict_reader(csv_data):
    reader = csv.DictReader(csv_data)
    return reader

def save_to_disk(data,file_name):
    """
    A function to save any data structure to a file using pickle module
    :param data: data structure that needs to be saved to disk
    :param file_name: name of the file to be used for saving the data
    :return: null
    """
    fileObject = open(file_name,'wb')
    pickle.dump(data,fileObject)
    fileObject.close()


def load_from_disk(file_name):
    """
    A function to load any data structure from a file using pickle module
    :param file_name: name of the file to be used for saving the data
    :return: data structure that exists in the file
    """
    fileObject = open(file_name,'r')
    data = pickle.load(fileObject)
    fileObject.close()
    return data