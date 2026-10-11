"""Run every question in homework1.py and print the results.

Expects fantasy_10000.json.gz and beer_50000.json in this folder.
Usage: python run_hw1.py
"""
import ast
import gzip
import json
import random
import warnings
from pathlib import Path

import dateutil.parser

import homework1

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

# Goodreads fantasy reviews (Q1-Q4)
with gzip.open(HERE / "fantasy_10000.json.gz") as f:
    books = [json.loads(line) for line in f]
for d in books:
    d["parsed_date"] = dateutil.parser.parse(d["date_added"])

shuffled = books[:]
random.seed(0)
random.shuffle(shuffled)

# BeerAdvocate reviews (Q5-Q7); one Python dict literal per line
with open(HERE / "beer_50000.json") as f:
    beers = [ast.literal_eval(line) for line in f]

print(f"Goodreads reviews: {len(books)}, beer reviews: {len(beers)}\n")

theta, mse1 = homework1.Q1(books)
print(f"Q1 theta = {theta}, MSE = {mse1:.4f}")
print(f"Q2 MSE = {homework1.Q2(books)[2]:.4f}")
print(f"Q3 MSE = {homework1.Q3(books)[2]:.4f}")
test2, test3 = homework1.Q4(shuffled)
print(f"Q4 test MSE: one-hot = {test2:.4f}, direct = {test3:.4f}")

TP, TN, FP, FN, ber5 = homework1.Q5(beers, homework1.featureQ5)
print(f"Q5 TP={TP} TN={TN} FP={FP} FN={FN} BER = {ber5:.4f}")
print("Q6 precision@[1, 100, 1000, 10000] =", homework1.Q6(beers))
ber7 = homework1.Q5(beers, homework1.featureQ7)[4]
print(f"Q7 BER (length) = {ber5:.4f}, BER (length + style) = {ber7:.4f}")
