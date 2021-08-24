# Predicting-Depression-Symptoms
Predicting depression symptoms using chat messages to bootstrap AI Chatbots.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
    </li>
    <li>
      <a href="#import-libraries">Import Libraries</a>
    </li>
    <li><a href="#data-collection">Data Collection</a></li>
    <li><a href="#eda">EDA</a></li>
    <li><a href="#model-architecture">Model Architecture</a></li>
    <li><a href="#model-performance">Model Performance</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- OVERVIEW -->
## Overview
Depression is a chronic mental illness that afflicts several millions each year. Mental health professionals rely on standard questionnaire like symptoms of Depression Questionnaire SDQ to identify the symptoms. SDQ leverages a Likert 5 scale of various questions to identify the symptoms like anxiety, anger, irritability etc. that help identify depression. Pedrelli has attempted to expand SDQ with several more questions due to its initial limitation in the structure of the SDQ questions. However, with the advent of machine learning and big data, we can leverage other sources of data like social media, chats and chatbots to form a more complete opinion of the symptoms of depression. This will assist mental health professionals with a more complete picture of symptoms and avoid the inherent bias in symptoms reporting due to stigma in self reporting such conditions in questionnaires. 

This research focuses on the following questions: 

•	How can one get a complete picture of depression symptoms?

•	How social media text and conversations can help with depression symptoms? 

•	What machine learning (ML) techniques are useful in diagnosing depression?

•	How can intelligent chatbots assist mental health professionals with diagnosis?

The main contributions of this research are: 

•	Evaluate depression symptoms using Machine Learning techniques

•	Create a novel method to analyze depression symptoms in popular chat engines for young adults

•	Build a machine learning model to help power an intelligent chatbot to assist mental health  
professionals for diagnosing depression



<!-- Import Libraries -->
## Import Libraries
To run the code, open the Symptom_Research.ipynb file. 

```
# data manipulation
import pandas as pd
import numpy as np

# plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# natural language processing
import re
import nltk
from nltk.tokenize import TweetTokenizer
import string
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# model building 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# neural network 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential

# model evaluation
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
```

<!-- DATA COLLECTION-->
## Data Collection
Discord is a chat tool that is widely used by young adults especially gamers. Discord instant messaging and digital distribution platform is designed for creating young communities. Along with text messaging, users communicate with voice and video calls as well. The public chats are communities of so called “servers”. Servers are a collection of persistent chat rooms. In discord, the channel #depression is a widely used section for expressing and discussing depression. We downloaded 23,995 samples of posts and their comments by various users, ranging from 2021-04-20 to 2021-05-20. We performed cleanup and preprocessing of the chats to map a user’s sentences to one or more symptoms of depression.

The data contains the following attributes: 
ID, AuthorID, Date, Chat Content, Word Count, and indicators for the seven target depression symptoms and no depression:

1. Change in appetite, losing or gaining weight  
2. Sleeping too much or not sleeping well (insomnia)
3. Fatigue and low energy most days
4. Feeling worthless, guilty, and hopeless
5. An inability to focus and concentrate that may interfere with daily tasks at home, work, or school
6. Movements that are unusually slow or agitated (a change which is often noticeable to others)
7. Thinking about death and dying; suicidal ideation or suicide attempts
8. None

<!-- EDA -->
## EDA
Distribution of text length in user messages

<img src="https://user-images.githubusercontent.com/35647593/130553587-05c1a65a-533c-411b-a670-563c69469d4e.png" width="510" height="300" />

Top N words 

<img src="https://user-images.githubusercontent.com/35647593/130553778-8968beb8-d6bc-4f56-8bf2-7d59c29854c5.png" width="510" height="450" />

Word Cloud Top Words

<img src="https://user-images.githubusercontent.com/35647593/130553885-4845d7d5-a3da-4ec4-9bcd-03a95460a435.png" width="510" height="500" />

Topics

<img src="https://user-images.githubusercontent.com/35647593/130553996-a4fbf81c-9df9-4e57-93c4-b177b6a089bb.png" width="510" height="510" />


<!-- MODEL ARCHITECTURE -->
## Model Architecture

Model Framework

<img src="https://user-images.githubusercontent.com/35647593/130554558-58f5ba74-3061-4af6-b51a-78689454cae6.JPG" width="510" height="300" />

This figure shows the Model training design of the Depression predictor. The discord chat server data is collected and is stored securely in a cloud behind a secure layer. The Chat data from depression channel is analyzed by a Symptom Extraction layer (EEL). The SEL is an asynchronous process that consists of training models such as Random Forest (RF), Logistic Support Vector Machine (SVM), Naïve Bayes (NB) and Convolution Neural Network (CNN). The SEL results can be symptom predictions that can be aggregated by medical professionals to predict depression.

<!-- MODEL PERFORMANCE -->
## Model Performance

For the final depression prediction, we built an ensemble model based on the individual symptom classifiers. For each data point, we parsed it into each of the seven symptom classifiers and recorded the prediction results. Then we set 6 thresholds, 1-6, for depression labelling. If the number of symptom models that returned positive prediction exceeded a certain threshold, then we labeled the data point as depression; otherwise, we label it as not depression. 

The final model was evaluated on the 20% validation set, and the performance metrics for each threshold are shown below:

|  Threshold	 |  Recall  |  Accuracy  |	 AUC   |
| ------------ | -------- | ---------- | ------- |
|>=1 symptoms	 |  0.6968	|   0.6286	 |  0.6488 |
|>=2 symptoms	 |  0.4579	|   0.7973   |  0.6422 |
|>=3 symptoms	 |  0.4149	|   0.9274	 |  0.6763 |
|>=4 symptoms	 |  0.3077	|   0.9746	 |  0.6429 |
|**>=5 symptoms**	 |  **0.5000**	|  **0.9954**	 |  **0.7479** |


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
[1] Anon, 2019. Investigators from University of York Have Reported New Data on Mental Health Diseases and Conditions (Predicting Persistent Depressive Symptoms In Older Adults: a Machine Learning Approach To Personalised Mental Healthcare). Mental Health Weekly Digest, p.158. 

[2] Zheng, F., Zhong, B., Song, X., & Xie, W. (2018). Persistent depressive symptoms and cognitive decline in older adults. The British Journal of Psychiatry, 213(5), 638-644. doi:10.1192/bjp.2018.155

[3] Pedrelli, P., Blais, M., Alpert, J., Shelton, R., Walker, R., & Fava, M. (2014). Reliability and validity of the Symptoms of Depression Questionnaire (SDQ). CNS Spectrums, 19(6), 535-546. doi:10.1017/S1092852914000406

[4] Kellner, Robert. "A symptom questionnaire." J Clin Psychiatry 48.7 (1987): 268-274.

[5] Verma, Anirudh, Shashikant Tyagi, and Gauri Mathur. "A Comprehensive Review on Bot-Discord Bot." (2021), International Journal of Scientific Research in Computer Science, Engineering and Information Technology, 2021
