import os
import numpy as np 
import pandas as pd 


import urllib
import ntpath
import urllib.request
from pathlib import Path


data_csv_path = "../data/gldv2_info.csv"

df = pd.read_csv(data_csv_path, delimiter=',', encoding='utf8')


opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

# def path_leaf(path):
#     head, tail = ntpath.split(path)
#     return tail or ntpath.basename(head)

filename = "images"

# open file to read
with open(data_csv_path, 'r') as csvfile:
    # iterate on all lines
    i = 0
    for line in csvfile:
        splitted_line = line.split(',')
        img_filename = splitted_line[0]
        class_name = splitted_line[3]

        # check if we have an image URL
        if splitted_line[1] != '' and splitted_line[1] != "\n" and splitted_line[1] != "url":
            try:
              Path("../data/images/"+ class_name+'/').mkdir(parents=True, exist_ok=True)
              urllib.request.urlretrieve(splitted_line[1], "../data/images/"+ class_name+'/'+ img_filename+'.jpg')
              i += 1
            except:
              pass
        else:
          pass