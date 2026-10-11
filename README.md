# CSE 258: Recommender Systems

Coursework for CSE 258 at UC San Diego. Homework 1 builds regression and classification models on two review datasets (Goodreads and BeerAdvocate) using NumPy and scikit-learn.

## What's in here

| Folder | Contents |
|---|---|
| `hw1/` | Homework 1: predicting review ratings and sentiment from review text, length, date, and beer style |

## Homework 1

**Book ratings (10,000 Goodreads fantasy reviews), Q1-Q4**
- Fit a linear regression (least squares) that predicts a review's star rating from its length.
- Add the day of the week and month, first as one-hot features and then as plain numbers, and compare the two encodings.
- Compare both encodings on held-out data with a 50/50 train/test split.

**Beer reviews (50,000 BeerAdvocate reviews), Q5-Q7**
- Fit a logistic regression that predicts whether a review is positive (overall rating of 4 or higher) from review length, with class weights to handle the imbalance.
- Evaluate with balanced error rate (BER, the average of the false-positive and false-negative rates) and precision at K.
- Add the beer style as a one-hot feature and measure the change in BER.

### Results

| Question | Model | Result | Evaluated on |
|---|---|---|---|
| Q1 | Rating ~ review length | MSE 1.552 | all 10,000 reviews |
| Q2 | + weekday, month (one-hot) | MSE 1.547 | all 10,000 reviews |
| Q3 | + weekday, month (numeric) | MSE 1.552 | all 10,000 reviews |
| Q4 | Q2 vs Q3 features | test MSE 1.626 vs 1.628 | held-out 50% (shuffled, seed 0) |
| Q5 | Positive review ~ length | BER 0.468 | all 50,000 reviews |
| Q6 | Same model, ranked by score | precision@1 / 100 / 1000 / 10000 = 1.00 / 0.75 / 0.71 / 0.715 | all 50,000 reviews |
| Q7 | + beer style (one-hot) | BER 0.360 | all 50,000 reviews |

Numbers come from running `hw1/run_hw1.py` (Python 3.11, NumPy 1.24, scikit-learn 1.5). Review length and date explain very little of the star rating: adding date features changes MSE by about 0.005. Beer style helps the classifier more, lowering BER from 0.468 to 0.360. Q1-Q3 and Q5-Q7 are measured on the data the models were fit on, so they are not estimates of out-of-sample performance.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| Modeling | NumPy (`np.linalg.lstsq`), scikit-learn (`LogisticRegression`, `train_test_split`, `mean_squared_error`) |
| Data handling | `gzip`, `json`, `python-dateutil` |
| Notebooks | Jupyter (`hw1/hw1_stub.ipynb`) |

## Quick start

The datasets are provided by the course and are not included in this repository. Put `fantasy_10000.json.gz` and `beer_50000.json` in `hw1/`, then:

```bash
git clone https://github.com/arundhatisingh17/CSE-258-Recommender-Systems.git
cd CSE-258-Recommender-Systems/hw1
pip install numpy scikit-learn python-dateutil
python run_hw1.py
```

The script prints the result for each question (Q1-Q7).

## Project structure

```
hw1/
  homework1.py       # solutions for Q1-Q7 (feature functions and models)
  run_hw1.py         # runs all questions and prints results
  hw1_stub.ipynb     # notebook version of the same work
```

## Limitations

- Q1-Q3 and Q5-Q7 report error on the training data, with no held-out set. Only Q4 uses a train/test split.
- The Q7 features are only one-hot beer style plus review length.
- The data files are not in the repository, so the code cannot run until they are added.

## Note

This is my own coursework. If you are taking the course, please write your own solutions.
