# Logistic Regression - SOFTMAX (One-vs-Rest)

This project presents an implementation of a multiclass classifier based on logistic regression. The solution follows the **one-vs-rest** approach: a separate binary classifier is trained for each class, and their outputs are then combined and normalized to obtain the probability distribution of a sample belonging to each class.

The program was tested on the **Iris** dataset from `scikit-learn`, using two input features:
- `petal length`
- `petal width`

## Main idea

A standard binary logistic regression model decides whether a sample belongs to the positive class (`1`) or the negative class (`0`).

In this project, the multiclass problem is solved by training several binary classifiers:

- classifier for class `0` vs all other classes,
- classifier for class `1` vs all other classes,
- classifier for class `2` vs all other classes.

Each classifier returns the probability that a sample belongs to its own class. These values are then **normalized**, so that the sum of probabilities for a single sample is equal to `1`.

## Project structure

The project consists of two files:

- `reglog.py` - main implementation of logistic regression and the multiclass classifier,
- `plotka.py` - helper function for plotting decision regions.

## Classes and methods

### `LogisticRegressionGD`

This class implements **binary logistic regression** trained with gradient descent.

#### `__init__(eta, n_iter, random_state)`
Initializes the model parameters:
- `eta` - learning rate,
- `n_iter` - number of training iterations,
- `random_state` - random seed used for weight initialization.

#### `fit(X, y)`
Trains the binary classifier.

Inside this method:
1. the weights are initialized randomly,
2. the linear input `net_input` is computed,
3. the output is passed through the sigmoid activation function,
4. the prediction error is calculated,
5. the weights and bias are updated.

#### `net_input(X)`
Computes the linear combination:
`w0 + w1*x1 + w2*x2 + ...`

#### `activation(z)`
Sigmoid activation function:
`1 / (1 + e^(-z))`

`np.clip` is used to avoid numerical overflow for very large values.

#### `predict_proba(X)`
Returns the probability that a sample belongs to the positive class.

#### `predict(X)`
Returns the binary class:
- `1` if the linear output is greater than or equal to `0`,
- `0` otherwise.

---

### `OneVsRestLogisticRegression`

This class implements the **multiclass classifier** based on several binary logistic regression models.

#### `__init__(eta, n_iter, random_state)`
Stores the parameters that will be passed to every binary classifier.

#### `fit(X, y)`
This method:
1. extracts all unique classes from the dataset,
2. creates binary labels for each class:
   - `1` for the current class,
   - `0` for all remaining classes,
3. trains a separate `LogisticRegressionGD` model for each class,
4. stores all trained models in `classifiers_`.

#### `predict_proba(X)`
For each sample:
1. collects the probability returned by each binary classifier,
2. combines them into one matrix,
3. normalizes each row so that the probabilities sum to `1`.

This produces a normalized probability distribution across all classes.

#### `predict(X)`
Selects the class with the highest probability.

## Data preparation

The program uses the `Iris` dataset, but only two features are selected:
- petal length,
- petal width.

The data is split into:
- training set (`70%`)
- test set (`30%`)

using `train_test_split(..., stratify=y)`.

Then **manual standardization** is applied:
- the mean and standard deviation are computed on the training set,
- both `X_train` and `X_test` are scaled using these training statistics.

Standardization significantly improves the behavior of gradient descent, because logistic regression is more stable when the features are on a similar scale.

## Program workflow

The `main()` function performs the following steps:

1. loads the `Iris` dataset,
2. selects two input features,
3. splits the data into training and test sets,
4. standardizes the data,
5. creates a `OneVsRestLogisticRegression` model,
6. trains the model,
7. predicts labels for the test set,
8. computes accuracy,
9. plots the decision regions for the training set.

## Result

After standardization, the model achieves an accuracy of about:

`0.9778`

which indicates very good classification performance for this task.

## Required libraries

The program uses:
- `numpy`
- `matplotlib`
- `scikit-learn`

## Running the program

Example:

```bash
python reglog.py