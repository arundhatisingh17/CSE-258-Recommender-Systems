from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import gzip
import json
import dateutil.parser
import numpy as np
import random


### Question 1


def getMaxLen(dataset):
    # Find the longest review (number of characters)
    maxLen = 0
    for item in dataset:
        maxLen = max(maxLen, len(item['review_text']))
        
    return maxLen


def featureQ1(datum, maxLen):
    # Feature vector for one data point
    return [1, (len(datum["review_text"]) / maxLen)]


def Q1(dataset):
    # Implement...
    y = [d["rating"] for d in dataset]
    maxLen = getMaxLen(dataset)

    X = [featureQ1(d, maxLen) for d in dataset]
    theta, residual, rank, s = np.linalg.lstsq(X, y)

    predictions = np.array(X) @ theta
    MSE = mean_squared_error(y, predictions)

    return theta, MSE


### Question 2


def featureQ2(datum, maxLen):
    # Implement (should be 1, length feature, day feature, month feature)

    timestamp = datum["parsed_date"]
    day = [0] * 6
    if timestamp.weekday() > 0:
        day[timestamp.weekday() - 1] = 1

    month = [0] * 11
    if timestamp.month > 1:
        month[timestamp.month - 2] = 1

    return [1, len(datum['review_text']) / maxLen] + day + month


def Q2(dataset):

    maxLen = getMaxLen(dataset)

    # Implement (note MSE should be a *number*, not e.g. an array of length 1)
    X2 = [featureQ2(d, maxLen) for d in dataset]
    Y2 = [d["rating"] for d in dataset]
    theta, residual, rank, s = np.linalg.lstsq(X2, Y2)

    predictions = np.array(X2) @ theta
    MSE2 = mean_squared_error(Y2, predictions)

    return X2, Y2, MSE2


### Question 3


def featureQ3(datum, maxLen):
    # Implement
    timestamp = datum["parsed_date"]
    return [1, len(datum["review_text"]) / maxLen, timestamp.weekday(), timestamp.month]


def Q3(dataset):

    maxLen = getMaxLen(dataset)

    X3 = [featureQ3(d, maxLen) for d in dataset]
    Y3 = [d["rating"] for d in dataset]
    theta, residual, rank, s = np.linalg.lstsq(X3, Y3)

    predictions = np.array(X3) @ theta
    MSE3 = mean_squared_error(Y3, predictions)

    return X3, Y3, MSE3


### Question 4


def Q4(dataset):

    # Implement
    # The runner shuffles the data before calling Q4, so don't shuffle again here
    maxLen = getMaxLen(dataset)
    train, test = train_test_split(dataset, test_size=0.5, shuffle=False)

    Y_train = [d["rating"] for d in train]
    Y_test = [d["rating"] for d in test]

    # One-hot encoding (Question 2)
    X2_train = [featureQ2(d, maxLen) for d in train]
    X2_test = [featureQ2(d, maxLen) for d in test]
    theta2, residual, rank, s = np.linalg.lstsq(X2_train, Y_train)
    test_mse2 = mean_squared_error(Y_test, np.array(X2_test) @ theta2)

    # Direct encoding (Question 3)
    X3_train = [featureQ3(d, maxLen) for d in train]
    X3_test = [featureQ3(d, maxLen) for d in test]
    theta3, residual, rank, s = np.linalg.lstsq(X3_train, Y_train)
    test_mse3 = mean_squared_error(Y_test, np.array(X3_test) @ theta3)

    return test_mse2, test_mse3


### Question 5


def featureQ5(datum):
    # Implement
    return [1, len(datum["review/text"])]


def Q5(dataset, feat_func):

    # Implement
    X = [feat_func(d) for d in dataset]
    y = [d["review/overall"] >= 4 for d in dataset]

    mod2 = linear_model.LogisticRegression(class_weight = 'balanced')
    mod2.fit(X, y)
    pred = mod2.predict(X)

    TP = sum(np.logical_and(pred, y))
    FP = sum(np.logical_and(pred, np.logical_not(y)))
    TN = sum(np.logical_and(np.logical_not(pred), np.logical_not(y)))
    FN = sum(np.logical_and(np.logical_not(pred), y))

    BER = 1 - 0.5*(TP / (TP + FN) + TN / (TN + FP))

    return TP, TN, FP, FN, BER


### Question 6


def Q6(dataset):
    # Implement
    X = [featureQ5(d) for d in dataset]
    y = [d["review/overall"] >= 4 for d in dataset]

    mod2 = linear_model.LogisticRegression(class_weight = 'balanced')
    mod2.fit(X, y)
    pred = mod2.predict(X)
    scores = mod2.decision_function(X)

    scoreslabels = list(zip(scores, y))
    scoreslabels.sort(reverse=True)
    sortedlabels = [x[1] for x in scoreslabels]

    precs = [sum(sortedlabels[:K]) / K for K in [1, 100, 1000, 10000]]

    return precs


### Question 7


# Implement (any feature vector which improves performance over Q5)
styles = [
    "Altbier",
    "American Adjunct Lager",
    "American Amber / Red Ale",
    "American Amber / Red Lager",
    "American Barleywine",
    "American Black Ale",
    "American Blonde Ale",
    "American Brown Ale",
    "American Dark Wheat Ale",
    "American Double / Imperial IPA",
    "American Double / Imperial Pilsner",
    "American Double / Imperial Stout",
    "American IPA",
    "American Malt Liquor",
    "American Pale Ale (APA)",
    "American Pale Lager",
    "American Pale Wheat Ale",
    "American Porter",
    "American Stout",
    "American Strong Ale",
    "American Wild Ale",
    "Baltic Porter",
    "Belgian Dark Ale",
    "Belgian IPA",
    "Belgian Pale Ale",
    "Belgian Strong Dark Ale",
    "Belgian Strong Pale Ale",
    "Berliner Weissbier",
    "Bi\u00c3\u00a8re de Garde",
    "Black & Tan",
    "Bock",
    "Braggot",
    "California Common / Steam Beer",
    "Chile Beer",
    "Cream Ale",
    "Czech Pilsener",
    "Doppelbock",
    "Dortmunder / Export Lager",
    "Dubbel",
    "Dunkelweizen",
    "Eisbock",
    "English Barleywine",
    "English Bitter",
    "English Brown Ale",
    "English Dark Mild Ale",
    "English India Pale Ale (IPA)",
    "English Pale Ale",
    "English Pale Mild Ale",
    "English Porter",
    "English Stout",
    "English Strong Ale",
    "Euro Dark Lager",
    "Euro Pale Lager",
    "Euro Strong Lager",
    "Extra Special / Strong Bitter (ESB)",
    "Flanders Oud Bruin",
    "Flanders Red Ale",
    "Foreign / Export Stout",
    "Fruit / Vegetable Beer",
    "German Pilsener",
    "Hefeweizen",
    "Herbed / Spiced Beer",
    "Irish Dry Stout",
    "Irish Red Ale",
    "Keller Bier / Zwickel Bier",
    "Kristalweizen",
    "K\u00c3\u00b6lsch",
    "Lambic - Fruit",
    "Lambic - Unblended",
    "Light Lager",
    "Low Alcohol Beer",
    "Maibock / Helles Bock",
    "Milk / Sweet Stout",
    "Munich Dunkel Lager",
    "Munich Helles Lager",
    "M\u00c3\u00a4rzen / Oktoberfest",
    "Oatmeal Stout",
    "Old Ale",
    "Pumpkin Ale",
    "Quadrupel (Quad)",
    "Rauchbier",
    "Russian Imperial Stout",
    "Rye Beer",
    "Saison / Farmhouse Ale",
    "Schwarzbier",
    "Scotch Ale / Wee Heavy",
    "Scottish Ale",
    "Scottish Gruit / Ancient Herbed Ale",
    "Smoked Beer",
    "Tripel",
    "Vienna Lager",
    "Weizenbock",
    "Wheatwine",
    "Winter Warmer",
    "Witbier",
]
styleIndex = {s : i for i, s in enumerate(styles)}

def featureQ7(datum):

    oneHot = [0] * len(styles)
    if datum["beer/style"] in styleIndex:
        oneHot[styleIndex[datum["beer/style"]]] = 1

    return [1, len(datum["review/text"])] + oneHot