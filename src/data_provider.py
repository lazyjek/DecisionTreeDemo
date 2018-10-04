#-*- coding: utf-8 -*-
import arff

def data_provider(filename):
    data = arff.load(open(filename, 'rb'))
    # return feature and datasets
    return data['attributes'], data['data']

