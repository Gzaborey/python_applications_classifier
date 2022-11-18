"""spam_detector

This tool allows user to make an estimation about whether the given request is a spam or not.
Models are trained in advance, however it is possible to retrain model on a new data using 'train'
function.

This tool requires Pandas, Tensorflow, Scikit Learn, NLTK be installed within the Python environment
where you are using this tool.
"""

import pandas as pd
import string
import re
import nltk
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model


nltk.download('stopwords')


def process_text(text):
    """Removes punctuation, whitespaces, stopwords, english words, short words and digits.
    Returns list of words."""

    stemmer = SnowballStemmer('russian')
    eng_regex = re.compile(r'[a-zA-Z]')

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    processed_words = [stemmer.stem(word.lower()) for word in nopunc.split()
                       if word.lower() not in stopwords.words('russian')
                       and word.lower() not in stopwords.words('english')
                       and word.lower().isalpha()
                       and len(word.lower()) > 4
                       and word.lower() not in ''.join(eng_regex.findall(word.lower()))]
    return processed_words


def train(x, y):
    """Takes Pandas Dataframe as 'X' and Pandas series as 'y' as inputs. Trains classifier and stores it in
    the SpamClassifier package for future use."""

    email_dom_vectorizer = CountVectorizer()
    email_dom_vectorizer.fit(x.email_domain)
    email_dom_vectorized_train = email_dom_vectorizer.transform(x.email_domain)

    email_message_vectorizer = CountVectorizer(analyzer=process_text)
    email_message_vectorizer.fit(x.message)
    email_message_vectorized_train = email_message_vectorizer.transform(x.message).toarray()

    nb_model = MultinomialNB()
    nb_model.fit(email_dom_vectorized_train, y)
    nb_predictions = nb_model.predict_proba(email_dom_vectorized_train)[:, 1]

    ann_model = Sequential()
    ann_model.add(Dense(128, input_dim=email_message_vectorized_train.shape[1], activation='relu'))
    ann_model.add(Dense(512, activation='relu'))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(128, activation='relu'))
    ann_model.add(Dense(64, activation='relu'))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(32, activation='relu'))
    ann_model.add(Dense(16, activation='relu'))
    ann_model.add(Dropout(0.2))
    ann_model.add(Dense(8, activation='relu'))
    ann_model.add(Dense(1, activation='sigmoid'))

    ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ann_model.fit(email_message_vectorized_train, y, epochs=10, batch_size=20, verbose=0)
    ann_predictions_proba = ann_model.predict(email_message_vectorized_train)

    temp_df = x.loc[:, ['valid_first_name', 'valid_phone_number']]
    temp_df = pd.concat([temp_df, pd.Series(ann_predictions_proba.reshape(ann_predictions_proba.shape[0],))], axis=1)
    temp_df = pd.concat([temp_df, pd.Series(nb_predictions.reshape(nb_predictions.shape[0],))], axis=1)

    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(temp_df, y)

    ann_file_name = 'SpamClassifier/trained_ann'
    ann_model.save(ann_file_name)

    nb_file_name = 'SpamClassifier/trained_nb.sav'
    joblib.dump(nb_model, nb_file_name)

    lr_file_name = 'SpamClassifier/trained_lr.sav'
    joblib.dump(lr_model, lr_file_name)

    dom_vectorizer_file_name = 'SpamClassifier/dom_vec.sav'
    joblib.dump(email_dom_vectorizer, dom_vectorizer_file_name)

    email_message_vectorizer_file_name = 'SpamClassifier/mess_vec.sav'
    joblib.dump(email_message_vectorizer, email_message_vectorizer_file_name)


def predict(x):
    """Takes Pandas Dataframe as input. Based on trained classifier, predicts if given examples are spam.
    Outputs Pandas Series."""

    ann_model = load_model('SpamClassifier/trained_ann')
    nb_model = joblib.load('SpamClassifier/trained_nb.sav')
    lr_model = joblib.load('SpamClassifier/trained_lr.sav')
    email_dom_vectorizer = joblib.load('SpamClassifier/dom_vec.sav')
    email_message_vectorizer = joblib.load('SpamClassifier/mess_vec.sav')

    email_dom_vectorized_prod = email_dom_vectorizer.transform(x.email_domain)
    email_message_vectorized_prod = email_message_vectorizer.transform(x.message).toarray()

    nb_predictions = nb_model.predict_proba(email_dom_vectorized_prod)[:, 1]
    ann_predictions = ann_model.predict(email_message_vectorized_prod)

    temp_df = x.loc[:, ['valid_first_name', 'valid_phone_number']]
    temp_df = temp_df.reset_index(drop=True)
    temp_df = pd.concat([temp_df, pd.Series(ann_predictions.reshape(ann_predictions.shape[0], ))], axis=1)
    temp_df = pd.concat([temp_df, pd.Series(nb_predictions.reshape(nb_predictions.shape[0], ))], axis=1)

    lr_predictions = lr_model.predict(temp_df)

    return lr_predictions
