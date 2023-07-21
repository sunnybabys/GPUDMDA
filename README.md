# Overview
GPUDMDA : Predicting potential microbe-disease associations with graph attention autoencoder, positive-unlabeled learning and deep neural network
## Data
Data is available at [HMDAD](http://www.cuilab.cn/hmdad) and [Disbiome](https://disbiome.ugent.be/).  In this work，HMDAD is data1 and Disbiome is data2.

## Important document 
 - feature.py：This is used to learn the features of microbes and diseases from the  similarity networks .
 - k-means.py ：This is used to cluster positive samples, calculate the distance between each positive sample and the center of the class, and determine the number of spy samples.
 - xgboot.py：This is used to screen for reliable negative samples.
 - CV_123.py：This is used for MDAs classification.

## Environment
Install python3.7 for running this model. And these packages should be satisfied:

 - tensorflow-gpu $\approx$ 2.4.0
 - pytorch $\approx$ 1.12.1+cu116
 - xgboost $\approx$ 1.6.2
 - numpy $\approx$ 1.19.5
 - pandas $\approx$ 1.3.5
 - scikit-learn $\approx$ 1.0.2
 - matplotlib $\approx$ 3.5.2

               

## Usage
Taking HMDAD as an example，default is 5-fold cross validation on microbe-Disease pairs，to run the model：
```
python data1/CV_123.py
```

The variable "cv" in the “CV. getcv()” function:
 - “1” represents 5-fold cross validation on diseases.
 - “2” represents 5-fold cross validation on  microbes.
 - “3” represents 5-fold cross validation on  microbe-disease pairs.
