import Models

# load packages
import pandas as pd
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import pickle
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

from keras.utils import np_utils
# from sklearn import metrics

# set random seeds
np.random.seed(1)
# tf.random.set_seed(2)

def prepare_x(data):
    df1 = data[:40, :].T   # .T indicates transpose of the matrix
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    dataX = dataX.swapaxes(1,2)

    return dataX, dataY

# dec_train = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt')
# dec_test1 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt')
# dec_test2 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')
# dec_test3 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')
# dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
# del dec_test1
# del dec_test2
# del dec_test3
# gc.collect()


dec_train = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_8.txt')
dec_test2 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')
dec_test3 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')
dec_test = np.hstack((dec_test2, dec_test3))
del dec_test2
del dec_test3
gc.collect()

# extract limit order book data from the FI-2010 dataset
train_lob = prepare_x(dec_train)
test_lob = prepare_x(dec_test)

# extract label from the FI-2010 dataset
train_label = get_label(dec_train)
test_label = get_label(dec_test)

# prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon. 
trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T=10)
print(trainX_CNN.shape)
trainY_CNN = trainY_CNN[:,3] - 1
trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)

# prepare test data.
testX_CNN, testY_CNN = data_classification(test_lob, test_label, T=10)
testY_CNN = testY_CNN[:,3] - 1
testY_CNN = np_utils.to_categorical(testY_CNN, 3)


# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40,10], [60,10], [120,5], [3,1]]

# get Bilinear model
projection_regularizer = None
projection_constraint = keras.constraints.max_norm(3.0,axis=0)
attention_regularizer = None
attention_constraint = keras.constraints.max_norm(5.0, axis=1)
dropout = 0.1

 
model = Models.TABL(template, dropout, projection_regularizer, projection_constraint,
                    attention_regularizer, attention_constraint)
model.summary()

# create class weight
class_weight = {0 : 1e6/300.0,
                1 : 1e6/400.0,
                2 : 1e6/300.0}


# training
model.fit(trainX_CNN, trainY_CNN, batch_size=64, epochs=100, class_weight=class_weight)
#model.save('tabl_model.h5')

# result = model.evaluate(testX_CNN, testY_CNN,batch_size=64)
# print("test loss, test acc:", result)

from sklearn.metrics import classification_report

y_pred = model.predict(testX_CNN, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

round_testy = np.argmax(testY_CNN, axis=1)

print(classification_report(round_testy, y_pred_bool))

