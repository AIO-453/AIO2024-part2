import numpy as np
import pandas as pd

import re
import nltk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import TweetTokenizer
from collections import defaultdict

import CustomSoftmaxRegression as sm


# a) Xây dựng hàm chuẩn hóa văn bản
def test_normalize(text):
    text = str(text)
    # Lowercase
    text = text.lower()
    # Retweet old arcconym "RT" removal
    text = re.sub(r'^RT[\s]+', '', text)

    # Hyperlinks removal
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # Hashtags removal
    text = re.sub(r'#', '', text)

    # Punctuation removal
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    tokenizer = TweetTokenizer(preserve_case=False,
                               strip_handles=True,
                               reduce_len=True)
    text_tokens = tokenizer.tokenize(text)

    return text_tokens

# b. Xây dựng bô lưu giữ tần suất xuất hiện của các từ


def get_freqs(df):
    freqs = defaultdict(lambda: 0)
    for idx, row in df.iterrows():
        tweet = row['clean_text']
        label = row['category']
        for word in test_normalize(tweet):
            pair = (word, label)
            freqs[pair] += 1
    return freqs

# c. Xây dựng hàm vector đặc trưng


def get_feature(text, freqs):
    tokens = test_normalize(text)
    X = np.zeros(3)
    X[0] = 1

    for token in tokens:
        X[1] += freqs[(token, 0)]
        X[2] += freqs[(token, 1)]

    return X


if __name__ == "__main__":
    dataset_path = './Twitter_Data.csv'
    df = pd.read_csv(dataset_path)
    # print(df.head())

    # d. Trích xuất đặc trưng toàn bộ dữ liệu
    X = []
    y = []

    freqs = get_freqs(df)
    for idx, row in df.iterrows():
        tweet = row['clean_text']
        label = row['category']

        X_i = get_feature(tweet, freqs)
        X.append(X_i)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    n_classes = df['category'].nunique()
    n_samples = df['category'].size

    # Convert 'category' column to numerical labels if necessary
    # Assuming 'category' column contains strings like 'Positive', 'Negative', 'Neutral'
    # Ensure labels are within the range 0 to n_classes - 1
    label_mapping = {label: i for i, label in enumerate(
        sorted(df['category'].unique()))}
    df['category_encoded'] = df['category'].map(label_mapping)

    y = df['category_encoded'].to_numpy()
    y = y.astype(np.uint8)

    # Check for and handle values in 'y' outside the expected range
    # Clip values to be within 0 to n_classes - 1
    y = np.clip(y, 0, n_classes - 1)

    y_encoded = np.zeros((n_samples, n_classes), dtype=np.uint8)
    # Now 'y' values will be within the valid range for indexing.
    y_encoded[np.arange(n_samples), y] = 1

    val_size = 0.2
    test_size = 0.125
    random_state = 2
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded,
                                                      test_size=val_size,
                                                      random_state=random_state,
                                                      shuffle=is_shuffle)

    _train, X_test, _train, y_test = train_test_split(X_val, y_val,
                                                      test_size=test_size,
                                                      random_state=random_state,
                                                      shuffle=is_shuffle)

    normalizer = StandardScaler()
    X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
    X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
    X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

    lr = 0.01
    epochs = 200
    batch_size = 128

    np.random.seed(random_state)

    sofmax_regression = sm.CustomSoftmaxRegression(
        X_train, X_val, y_train, y_val, lr, epochs, batch_size)
    sofmax_regression.fit()

    val_set_acc = sofmax_regression.compute_accuracy(X_val, y_val)
    test_set_acc = sofmax_regression.compute_accuracy(X_test, y_test)
    print('Evaluation on validation and test set :')
    print(f'Accuracy : {val_set_acc}')
    print(f'Accuracy : {test_set_acc}')

    sofmax_regression.plot()
