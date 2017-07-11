from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import exp

def load_data(filename):
    # load csv file
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for line in csv_reader:
            if not line:
                continue
            dataset.append(line)
    return dataset

def convert_to_float(dataset):
    # convert string to float
    data = list()
    for row in dataset:
        cols = row[:-1] + [row[-1].rstrip('\\\\')]
        vals = [float(c.strip()) for c in cols]
        data.append(vals)
    return data

def min_max(dataset):
    # Find the min and max values for each column
    minmax = list()
    for i in xrange(len(dataset[0])):
        col_val = [row[i] for row in dataset]
        min_val = min(col_val)
        max_val = max(col_val)
        minmax.append([min_val, max_val])
    return minmax

def normalize(dataset, minmax):
    # Rescale dataset columns to the range 0-1
    for row in dataset:
        for i in xrange(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def n_fold(dataset, n):
    # Split a dataset into k folds
    dataset_copy = list(dataset)
    dataset_folds = list()
    size = int(len(dataset_copy) / n)
    for i in xrange(n):
        fold = list()
        while len(fold) < size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_folds.append(fold)
    return dataset_folds

def accuracy(pred, actual):
    n_sample = len(pred)
    cnt = 0
    for i in xrange(n_sample):
        if round(pred[i]) == actual[i]:
            cnt += 1
    return cnt / float(n_sample) * 100

def predit(X, coeficients):
    s = coeficients[0]
    for i in xrange(1, len(coeficients)):
        s += X[i-1] * coeficients[i]
    return 1.0 / (1.0 + exp(-s))

def sgd(train, epoch, rate):
    n_sample = len(train)
    X = [row[:-1] for row in train]
    y = [row[-1] for row in train]
    n_x_dim = len(X[0])
    coeficients = [0.0 for _ in xrange(n_x_dim + 1)]

    for e in xrange(epoch):
        s_error = 0
        for i in xrange(n_sample):
            y_pred = predit(X[i], coeficients)
            error = y_pred - y[i]
            s_error += error ** 2
            coeficients[0] -= rate * error
            for j in xrange(n_x_dim):
                coeficients[j + 1] -= rate * error * X[i][j]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (e, rate, s_error / n_sample))
    return coeficients

def bgd(train, epoch, rate):
    n_sample = len(train)
    X = [row[:-1] for row in train]
    y = [row[-1] for row in train]
    n_x_dim = len(X[0])
    coeficients = [0.0 for _ in xrange(n_x_dim + 1)]

    for e in xrange(epoch):
        s_error = 0
        c_error = 0
        for i in xrange(n_sample):
            y_pred = predit(X[i], coeficients)
            error = y_pred - y[i]
            c_error += error
            s_error += error ** 2
        m_error = c_error / n_sample
        coeficients[0] -= rate * m_error
        for j in xrange(n_x_dim):
            coeficients[j + 1] -= rate * m_error * X[i][j]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (e, rate, s_error / n_sample))
    return coeficients

def mgd(train, epoch, rate):
    n_sample = len(train)
    n_batch = 450
    batch_size = n_sample / n_batch
    X = [row[:-1] for row in train]
    y = [row[-1] for row in train]
    n_x_dim = len(X[0])
    coeficients = [0.0 for _ in xrange(n_x_dim + 1)]

    for e in xrange(epoch):
        s_error = 0
        for n in xrange(n_batch):
            b_error = 0
            start = n * batch_size
            end = (n + 1) * batch_size if n != n_batch-1 else n_sample
            for i in xrange(start, end):
                y_pred = predit(X[i], coeficients)
                error = y_pred - y[i]
                b_error += error
                s_error += error ** 2
            m_error = b_error / (end - start)
            coeficients[0] -= rate * m_error
            for j in xrange(n_x_dim):
                coeficients[j + 1] -= rate * m_error * X[i][j]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (e, rate, s_error / n_sample))
    return coeficients

def cross_validate(dataset, algorithm, n, *args):
    folds = n_fold(dataset, n)
    scores = list()
    for fold in folds:
        train = list(folds)
        train.remove(fold)
        train = sum(train, [])
        test = list(fold)
        coefficients = algorithm(train, *args)
        predictions = list()
        ground_truth = list()
        for i in xrange(len(test)):
            y_pred = predit(test[i][:-1], coefficients)
            predictions.append(y_pred)
            ground_truth.append(test[i][-1])
        scores.append(accuracy(predictions, ground_truth))
    return scores

seed(1)
filename = 'pima-indians-diabetes.csv'
dataset = load_data(filename)
dataset_float = convert_to_float(dataset)
minmax = min_max(dataset_float)
normalize(dataset_float, minmax)
n = 5
rate = 0.1
epoch = 50
scores = cross_validate(dataset_float, mgd, n, epoch, rate)

print('Scores: %s' % scores)
print('Mean accuracy: %.3f' % (sum(scores)/float(len(scores))))