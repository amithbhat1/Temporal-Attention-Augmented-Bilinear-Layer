# load packages
import pandas as pd
import pickle
import numpy as np
import gc
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D, CuDNNLSTM
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

from keras.utils import np_utils
import matplotlib.pyplot as plt

# set random seeds
from tensorflow import set_random_seed
from keras.backend.tensorflow_backend import set_session
np.random.seed(1)
set_random_seed(2)

# limit gpu usage for keras with tensorflow 1
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

def prepare_x(data):
    df1 = data[:40, :].T
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

    return dataX.reshape(dataX.shape + (1,)), dataY


# please change the data_path to your local path

dec_train = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_7.txt')
dec_test1 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_7.txt')
dec_test2 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_8.txt')
dec_test3 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_9.txt')
dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
del dec_test1
del dec_test2
del dec_test3
gc.collect()


# dec_train = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt')
# dec_test1 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt')
# dec_test2 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt')
# dec_test3 = np.loadtxt('/home/amithbn/Desktop/TABL/bench-data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')
# dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
# del dec_test1
# del dec_test2
# del dec_test3
# gc.collect()


# extract limit order book data from the FI-2010 dataset
train_lob = prepare_x(dec_train)
test_lob = prepare_x(dec_test)

# extract label from the FI-2010 dataset
train_label = get_label(dec_train)
test_label = get_label(dec_test)

# prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon. 
trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T=10)
trainY_CNN = trainY_CNN[:,3] - 1
trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)

# prepare test data.
testX_CNN, testY_CNN = data_classification(test_lob, test_label, T=10)
testY_CNN = testY_CNN[:,3] - 1
testY_CNN = np_utils.to_categorical(testY_CNN, 3)

def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))
    
    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)
    
    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    # If GPU is not available, change CuDNNLSTM to LSTM
    # conv_lstm = CuDNNLSTM(number_of_lstm)(conv_reshape)
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_deeplob(10, 40, 64)

model.fit(trainX_CNN, trainY_CNN, epochs=100, batch_size=1, verbose=2, validation_data=(testX_CNN, testY_CNN))

from sklearn.metrics import classification_report

y_pred = model.predict(testX_CNN, batch_size=1, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

round_testy = np.argmax(testY_CNN, axis=1)

print(classification_report(round_testy, y_pred_bool))
