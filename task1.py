import sys

sys.path.append("..")
import common.download_utils

common.download_utils.download_week1_resources()

from grader import Grader

grader = Grader()

import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')
train.head()

X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    word_tokens = nltk.word_tokenize(text)
    text = []
    for w in word_tokens:
        if w not in STOPWORDS:
            text.append(w)
    return ' '.join(text)


def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]

    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


print(test_text_prepare())

prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)

# print(text_prepare_results)
grader.submit_tag('TextPrepare', text_prepare_results)

X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]

X_train[:3]

# Dictionary of all tags from train corpus with their counts.
tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}

for x in X_train:
    words_train = nltk.word_tokenize(x)
    for w in words_train:
        if w in words_counts:
            words_counts[w] = words_counts.setdefault(w) + 1
        else:
            words_counts[w] = 1

for y in y_train:
    tags_train = REPLACE_BY_SPACE_RE.sub('', str(y))
    tags_train = tags_train.replace("'", '').split()
    for t in tags_train:
        if t in tags_counts:
            tags_counts[t] = tags_counts.setdefault(t) + 1
        else:
            tags_counts[t] = 1

most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

grader.submit_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags),
                                                ','.join(word for word, _ in most_common_words)))

DICT_SIZE = 5000
MOST_COMMON_WORDS = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE]
INDEX_TO_WORDS = {}
for i in range(DICT_SIZE):
    INDEX_TO_WORDS[i] = MOST_COMMON_WORDS[i][0]
# print(INDEX_TO_WORDS)
WORDS_TO_INDEX = {value: key for key, value in INDEX_TO_WORDS.items()}
# print(WORDS_TO_INDEX)
ALL_WORDS = WORDS_TO_INDEX.keys()


# print(ALL_WORDS)


def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    words = nltk.word_tokenize(text)
    index_to_words = {value: key for key, value in words_to_index.items()}
    for w in words:
        for i in range(dict_size):
            if w == index_to_words.get(i):
                result_vector[i] = 1
    return result_vector


def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


print(test_my_bag_of_words())

from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack(
    [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack(
    [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack(
    [sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)

row = X_train_mybag[10].toarray()[0]
# print(row)
non_zero_elements_count = 0
for i in range(DICT_SIZE):
    if row[i] == 1:
        non_zero_elements_count += 1
# print(non_zero_elements_count)
grader.submit_tag('BagOfWords', str(non_zero_elements_count))

from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(x_train, x_val, x_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')
    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_val = tfidf_vectorizer.transform(x_val)
    x_test = tfidf_vectorizer.transform(x_test)

    return x_train, x_val, x_test, tfidf_vectorizer.vocabulary_


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
# print(X_val_tfidf)
# print(X_train_tfidf)
# print(len(tfidf_vocab))
tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

for i in enumerate(tfidf_reversed_vocab):
    if tfidf_reversed_vocab[i[1]] == 'c#' or tfidf_reversed_vocab[i[1]] == 'c++':
        print("Good!")

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier


def train_classifier(x_train, Y_train):
    """
      X_train, y_train — training data

      return: trained classifier
    """
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    # C = [0.1, 1, 10, 100]
    # for c in C:
    trained_classifier = OneVsRestClassifier(LogisticRegression(C=10, max_iter=10000, penalty='l2')).fit(x_train,
                                                                                                         Y_train)
    return trained_classifier


classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)

y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


def print_evaluation_scores(y_val, predicted):
    print("Accuracy:", accuracy_score(y_val, predicted))
    print("F1 'macro'", f1_score(y_val, predicted, average='macro'))
    print("F1 'micro'", f1_score(y_val, predicted, average='micro'))
    print("F1 'weighted'", f1_score(y_val, predicted, average='weighted'))
    print("Average precision 'macro'", average_precision_score(y_val, predicted, average='macro'))
    print("Average precision 'micro'", average_precision_score(y_val, predicted, average='micro'))
    print("Average precision 'weighted'", average_precision_score(y_val, predicted, average='weighted'))


print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

test_predictions = classifier_tfidf.predict(X_test_tfidf)
test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
grader.submit_tag('MultilabelClassification', test_predictions_for_submission)


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary

        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    estimator = classifier.estimators_[tags_classes.index(tag)]

    top_positive_words = [index_to_words[index] for index in estimator.coef_.argsort().tolist()[0][-5:]]
    top_negative_words = [index_to_words[index] for index in estimator.coef_.argsort().tolist()[0][:5]]
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)

