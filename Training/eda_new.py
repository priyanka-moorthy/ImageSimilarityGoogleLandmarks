import os
import numpy as np 
import pandas as pd 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses

import matplotlib.pyplot as plt
import seaborn as sns
import warnings


import urllib
import ntpath
import urllib.request
from pathlib import Path


IMG_SIZE = (224, 224)
COLOR_MODE = "rgb"
COLOR_CHANNELS = 3 if COLOR_MODE == "rgb" else 1
BATCH_SIZE = 32 # since tensorflow doesn't load all the data into memory 
                #we need to define how many images are fetched together from memory


VALIDATION_SPLIT = 0.2
SEED = 42

data_path = '../data/images'
train_data = image_dataset_from_directory(data_path,
                                          color_mode=COLOR_MODE, 
                                          batch_size=BATCH_SIZE,
                                          image_size=IMG_SIZE,
                                          subset="training",
                                          seed=SEED,
                                          validation_split=VALIDATION_SPLIT)

val_data = image_dataset_from_directory(data_path,
                                        color_mode=COLOR_MODE, 
                                        batch_size=BATCH_SIZE,
                                        image_size=IMG_SIZE,
                                        subset="validation",
                                        seed=SEED,
                                        validation_split=VALIDATION_SPLIT)

label_map = {i:name for i,name in enumerate(train_data.class_names)}
CLASS_COUNT  = len(label_map)

label_counts = {label : len( os.listdir(f"{data_path}/{label}") ) for label in train_data.class_names }
label_counts = pd.Series(label_counts)
label_counts.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(12,12))
ax = sns.barplot(x=label_counts.values[:10], y=label_counts.index[:10])
ax.set_title("Top 10 Class counts")
plt.show()


#define figure and axes
fig, ax = plt.subplots()

#create values for table
table_data=[
    ["Training Loss", 0.012],
    ["Validation Loss", 0.011]
]

#create table
table = ax.table(cellText=table_data, loc='center')

#modify table
table.set_fontsize(14)
table.scale(1,4)
ax.axis('off')

#display table
plt.show()