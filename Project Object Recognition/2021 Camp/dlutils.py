import os, sys, time, datetime, fnmatch
import matplotlib.pyplot as plt
import numpy as np
import random
#from itertools import izip
from itertools import zip_longest as zip

# data normalizations (scaling down all values to the interval [0,1])
def unit_scale(X):
    a = X.min()
    b = X.max()
    X_scaled = (X - a) / float(b - a)
    return X_scaled

# Extract a balanced sample of n items from each class
# X = numpy array of data items
# Y = numpy array of features (usually integers)
def balanced_sample(X, Y, classes, n):
    d = {}
    for c in classes:
        d[c] = []
    for x,y in izip(X,Y):
        d[y].append((x,y))

    XY_samp = []
    for c in classes:
        if len(d[c]) < n:
            raise Exception("Sampling size %d exceeds class size %s" % (n,c))
        XY_samp.extend(random.sample(d[c], n))
    random.shuffle(XY_samp)
    X_samp, Y_samp = izip(*XY_samp)
    return np.asarray(X_samp), np.asarray(Y_samp)

def dlsplit(X, Y, N):
    M = len(X)
    a = np.arange(0,M)
    samp = np.random.choice(a,N)
    samp = np.random.choice(a, size=N, replace=False)
    samp.sort()
    X_train = X[samp]
    y_train = Y[samp]
    cosamp = np.setdiff1d(a,samp)
    X_test = X[cosamp]
    y_test = Y[cosamp]
    return X_train, y_train, X_test, y_test

def balance_classes(X, Y, nb_classes, size=None, csvfile=None):
    d = {}
    for y in range(nb_classes):
        d[y] = []
    for i,x in enumerate(X):
        y = Y[i]
        d[y].append(x)
    if size is None:
        size = max([len(d[y]) for y in d])
    e = {}
    for y in d:
        n = len(d[y])
        if n > size:
            e[y] = random.sample(d[y], size)
        elif 0<n<size:
            q, r = divmod(size, n)
            e[y] = q * d[y] + d[y][0:r]
        else:
            e[y] = d[y]
    X_bal = []
    y_bal = []
    for y in e:
        for x in e[y]:
           X_bal.append(x)
           y_bal.append(y)
    X_bal = np.array(X_bal)
    y_bal = np.array(y_bal)
    if csvfile is None:
        return X_bal, y_bal
    else:
        f = open(csvfile, 'w')
        pc = Progcount(size)
        for x,y in izip(X_bal, y_bal):
            v = np.append(x,y)
            f.write(','.join([str(i) for i in v]))
            f.write('\n')
            pc.advance()
        f.close()

def calc_class_weight(X, Y, nb_classes, e=1.0):
    d = {}
    for y in range(nb_classes):
        d[y] = 0
    for i,x in enumerate(X):
        y = Y[i]
        d[y] += 1
    cw = {}
    m = float(max(d.values()))
    for y in d:
        if d[y]:
            cw[y] = pow(m / d[y], e)
            #cw[y] = 1 + e * (m / d[y] - 1)
        else:
            cw[y] = 0
    return cw

def get_false_predictions(model, X, Y):
    y_pred = model.predict_classes(X)
    false_preds = [(x,y,p) for (x,y,p) in izip(X, Y, y_pred) if y != p]
    return y_pred, false_preds

def barchart(X, Y, nb_classes):
    d = {}
    for y in range(nb_classes):
        d[y] = 0
    for i,x in enumerate(X):
        d[Y[i]] += 1
    values = [d[y] for y in range(nb_classes)]
    plt.bar(range(nb_classes), values, align='center')
    plt.show()

def current_time(fmt='%Y-%m-%d %H:%M:%S'):
    t = datetime.datetime.strftime(datetime.datetime.now(), fmt)
    return t

def read_file(file):
    f=open(file,"r")
    text=f.read()
    f.close()
    return text

def write_file(file, data):
    f=open(file,"w")
    f.write(data)
    f.close()
    return file

def append_file(file, *data_args):
    f=open(file,"a+")
    for data in data_args:
        f.write(data)
    f.close()
    return file






