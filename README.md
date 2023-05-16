# ImageSimilarityGoogleLandmarks

Steps to reproduce the results

Download the data set from - https://github.com/cvdfoundation/google-landmark

The subset of dataset used is availabke on kaggle - https://www.kaggle.com/datasets/aymanmostafa11/eg-landmarks

- Create folder data/images

- Change to Training directory

- To Download the dataset run 
```
Python3 download_images.py
```


- To perform EDA run
```
Python3 eda_new.py
```


- Create folder ../models

- To train the auto encoder run
```
python3 train.py
```

- To run the webapp
```
python3 app.py
```

