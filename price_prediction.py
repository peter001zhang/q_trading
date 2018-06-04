import pandas as pd
import numpy as np
import os
from keras.models import load_model
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers import ConvLSTM2D
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.layers import Conv1D
from keras.layers import Multiply
from keras.layers import Input
from keras import Model
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn import svm
import xgboost as xgb
import data_processing

os.chdir(r'C:\cointree')
string=['aapl','amzn','blk','fb','googl','gs','jpm','msft','orcl','tsla']

class Price_prediction:
##read the data from the document
    def read_data(self,string):
        data = pd.read_csv('%s.us.txt'% string)
        return data

    ##change the data formation into the matrix
    def load_data(self,id):
        data = self.read_data(id)
        data = np.array(data)
        train_data = np.zeros((0,5))
        for row in data:
            cut_data = row[1:6]
            train_data = np.vstack((train_data,cut_data))
        return train_data




    def seperate_ML(self, name, p, win):  ###将数据拆分成训练集和测试集，并将win个一组摊平作为输出，同时进行归一化操作
        x = self.load_data(name)  ###股票名，训练集所占比例，窗口长度
        y = data_processing.close_price(name)
        r = len(x) // 10
        c = 0
        train_input = np.zeros((r * p, win, x.shape[1]))
        for i in range(0, r * p):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    t = x[i + j][u]
                    if x[i][u] != 0:
                        s = t / x[i][u]
                    else:
                        s = 1
                    s = s - 1
                    train_input[i][j][u] = s

        test_input = np.zeros((len(x) - 2 * win - r * p, win, x.shape[1]))
        for i in range(r * p + win, len(x) - win):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    t = x[i + j][u]
                    if x[i][u] != 0:
                        s = t / x[i][u]
                    else:
                        s = 1
                    s = s - 1
                    test_input[c][j][u] = s
            c = c + 1

        z = np.zeros((len(y), 1))
        for i in range(0, len(y) - win):
            z[i + win][0] = y[i + win][0] / y[i][0] - 1
        train_output = z[win:r * p + win]
        test_output = z[r * p + 2 * win:]
        trainx = np.zeros((len(train_input), win * x.shape[1]))
        testx = np.zeros((len(test_input), win * x.shape[1]))
        for i in range(0, len(trainx)):
            trainx[i] = train_input[i].flatten()
        for i in range(0, len(testx)):
            testx[i] = test_input[i].flatten()
        return trainx, testx, train_output, test_output


    def seperate_ML_general(self, name, data, p, win):  ###将数据拆分成训练集和测试集，并将win个一组摊平作为输出，同时进行归一化操作
        x = data  ###name为股票名称，data为用于拆分的数据，p为训练集所占比例乘10，win为窗口长度
        y = data_processing.close_price(name)
        r = len(x) // 10
        c = 0
        train_input = np.zeros((r * p, win, x.shape[1]))
        for i in range(0, r * p):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    t = x[i + j][u]
                    if x[i][u] != 0:
                        s = t / x[i][u]
                    else:
                        s = 1
                    s = s - 1
                    train_input[i][j][u] = s

        test_input = np.zeros((len(x) - 2 * win - r * p, win, x.shape[1]))
        for i in range(r * p + win, len(x) - win):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    t = x[i + j][u]
                    if x[i][u] != 0:
                        s = t / x[i][u]
                    else:
                        s = 1
                    s = s - 1
                    test_input[c][j][u] = s
            c = c + 1

        z = np.zeros((len(y), 1))
        for i in range(0, len(y) - win):
            z[i + win][0] = y[i + win][0] / y[i][0] - 1
        train_output = z[win:r * p + win]
        test_output = z[r * p + 2 * win:]
        trainx = np.zeros((len(train_input), win * x.shape[1]))
        testx = np.zeros((len(test_input), win * x.shape[1]))
        for i in range(0, len(trainx)):
            trainx[i] = train_input[i].flatten()
        for i in range(0, len(testx)):
            testx[i] = test_input[i].flatten()
        return trainx, testx, train_output, test_output


    def seperate_LSTM_normal(self,name, p, win):  ##将数据集拆分成训练和测试集，整理成3维，正规化后使之能够被lstm神经网络读取
        x = self.load_data(name)  ##股票名，训练集占比例，窗口长
        y = data_processing.close_price(name)
        r = len(x) // 10
        c = 0
        train_input = np.zeros((r * p, win, x.shape[1]))
        for i in range(0, r * p):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    t = x[i + j][u]
                    if x[i][u] != 0:
                        s = t / x[i][u]
                    else:
                        s = 1
                    s = s - 1
                    train_input[i][j][u] = s

        test_input = np.zeros((len(x) - 2 * win - r * p, win, x.shape[1]))
        for i in range(r * p + win, len(x) - win):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    t = x[i + j][u]
                    if x[i][u] != 0:
                        s = t / x[i][u]
                    else:
                        s = 1
                    s = s - 1
                    test_input[c][j][u] = s
            c = c + 1

        z = np.zeros((len(y), 1))
        for i in range(0, len(y) - win):
            z[i + win][0] = y[i + win][0] / y[i][0] - 1
        train_output = z[win:r * p + win]
        test_output = z[r * p + 2 * win:]

        return train_input, test_input, train_output, test_output


    def seperate_LSTM_general_normal(self,name, data, p, win):  ##输入数据，进行拆分，正规化
        x = data  ##data需要处理的数据
        y = data_processing.close_price(name)
        r = len(x) // 10
        c = 0
        train_input = np.zeros((r * p, win, x.shape[1]))
        for i in range(0, r * p):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    t = x[i + j][u]
                    if x[i][u] != 0:
                        s = t / x[i][u]
                    else:
                        s = 1
                    s = s - 1
                    train_input[i][j][u] = s

        test_input = np.zeros((len(x) - 2 * win - r * p, win, x.shape[1]))
        for i in range(r * p + win, len(x) - win):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    t = x[i + j][u]
                    if x[i][u] != 0:
                        s = t / x[i][u]
                    else:
                        s = 1
                    s = s - 1
                    test_input[c][j][u] = s
            c = c + 1

        z = np.zeros((len(y), 1))
        for i in range(0, len(y) - win):
            z[i + win][0] = y[i + win][0] / y[i][0] - 1
        train_output = z[win:r * p + win]
        test_output = z[r * p + 2 * win:]

        return train_input, test_input, train_output, test_output


    ### define a lstm_model，简单的lstm模型，128个神经元的lstm加128个神经元的全连接
    def LSTM_model(self,inputs, activ_func="linear",
                   dropout=0.10, loss="mae", optimizer="adam"):
        model = Sequential()

        model.add(LSTM(64, input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=128))
        model.add(Dense(units=1))
        model.add(Activation(activ_func))

        model.compile(loss=loss, optimizer=optimizer)

        return model


    ##gru模型，速度快，但是模型效果没有传统lstm好
    def GRU(self,traini, act="linear", dropout=0.10, loss="mae", optimizer="nadam"):
        model = Sequential()
        model.add(GRU(64, input_shape=(traini.shape[1], traini.shape[2]), return_sequences=True))
        model.add(GRU(128))
        model.add(Dropout(dropout))

        model.add(Dense(units=128))
        model.add(Dense(units=1))
        model.add(Activation(act))

        model.compile(loss=loss, optimizer=optimizer)

        return model


    ##卷积神经网络加lstm模型，deepsense模型
    def DEEPSENSE(self,traini, act="linear", dropout=0.1, loss="mae", optimizer="nadam"):
        model = Sequential()
        model.add(Conv2D(16, (2, traini.shape[2]), input_shape=(traini.shape[1], traini.shape[2], 1), padding='valid',
                         data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(Reshape((traini.shape[1] - 1, 16), input_shape=(traini.shape[1] - 1, 1, 16)))
        model.add(Conv1D(8, 3, input_shape=(traini.shape[1] - 1, 16)))
        model.add(Activation('relu'))
        model.add(Conv1D(4, 2))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(LSTM(128))
        model.add(Dropout(dropout))
        model.add(Dense(units=1))
        model.add(Activation(act))
        model.compile(loss=loss, optimizer=optimizer)

        return model


    ##加了图像识别中的注意力模型的CNN+LSTM
    def DEEPSENSE2(self,traini, act="linear", dropout=0.1, loss="mae", optimizer="nadam"):
        inputs = Input(shape=(traini.shape[1], traini.shape[2], traini.shape[3]))

        x = Conv2D(16, (2, traini.shape[2]), input_shape=(traini.shape[1], traini.shape[2], 1),
                   padding='valid', data_format='channels_last', activation='relu')(inputs)
        y1 = Conv2D(16, (2, traini.shape[2]), input_shape=(traini.shape[1], traini.shape[2], 1),
                    padding='valid', data_format='channels_last', activation='relu')(inputs)
        y2 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y1)
        y3 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y2)
        y = Conv2D(16, (1, 1), padding='valid', activation='relu')(y3)
        out = Multiply()([x, y])

        models = Model(inputs=inputs, outputs=out)

        model = Sequential()
        model.add(models)
        model.add(Reshape((traini.shape[1] - 1, 16), input_shape=(traini.shape[1] - 1, 1, 16)))
        model.add(Conv1D(8, 3, input_shape=(traini.shape[1] - 1, 16)))
        model.add(Activation('relu'))
        model.add(Conv1D(4, 2))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(LSTM(128))
        model.add(Dropout(dropout))
        model.add(Dense(units=1))
        model.add(Activation(act))
        model.compile(loss=loss, optimizer=optimizer)

        return model


    def xgboost_run(self,traini, traino, testi, testo):  ##xgboost回归模型
        dtrain = xgb.DMatrix(traini, label=traino)
        dtest = xgb.DMatrix(testi)

        bst = xgb.XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.4,
                               gamma=0, learning_rate=0.07, max_delta_step=0, max_depth=3,
                               min_child_weight=1.5, missing=None, n_estimators=10000, nthread=-1,
                               objective='reg:linear', reg_alpha=0.75, reg_lambda=0.45,
                               scale_pos_weight=1, seed=42, silent=True, subsample=0.6)
        bst.fit(traini, traino, verbose=True)
        pred = bst.predict(testi)
        plt.figure(figsize=(20, 5))
        plt.plot(testo, label="real data")
        plt.plot(pred, label="predicted data")
        plt.legend()
        plt.show()
        print(metrics.r2_score(testo, pred))


    def SVR_RUN(self,traini, traino, testi, testo):  ##svr模型
        clf = svm.SVR()
        clf.fit(traini, traino)
        pred = clf.predict(testi)
        print(metrics.r2_score(testo, pred))
        plt.figure(figsize=(20, 5))
        plt.plot(testo, label="real data")
        plt.plot(pred, label="predicted data")
        plt.legend()
        plt.show()


    #### define a run for lstm,LSTM的运行函数
    def LSTM_RUN(self, train_input, train_output, test_input, test_output, outputsize, neuron, epoch, batch):
        nn_model = self.LSTM_model(train_input)
        nn_history = nn_model.fit(train_input, train_output, epochs=epoch, batch_size=batch, verbose=2, shuffle=True)
        plt.figure(figsize=(20, 5))
        plt.plot(test_output, label="real data")
        plt.plot(nn_model.predict(test_input), label="predicted data")
        plt.legend()
        plt.show()
        MAE = mean_absolute_error(test_output, nn_model.predict(test_input))
        print(metrics.r2_score(test_output, nn_model.predict(test_input)))
        print(MAE)


    ##gru模型，速度快，但是模型效果没有传统lstm好
    def GRU_RUN(self,name, traini, traino, testi, testo, neurons, epoch, batch, act="linear", dropout=0.10, loss="mae",
                optimizer="nadam"):
        model = Sequential()
        model.add(GRU(neurons, input_shape=(traini.shape[1], traini.shape[2]), return_sequences=True))
        model.add(GRU(128))
        model.add(Dropout(dropout))

        model.add(Dense(units=128))
        model.add(Dense(units=1))
        model.add(Activation(act))

        model.compile(loss=loss, optimizer=optimizer)
        model.fit(traini, traino, epochs=epoch, batch_size=batch, verbose=2, shuffle=True)
        pred = model.predict(testi)
        print(metrics.r2_score(testo, pred))
        plt.figure(figsize=(20, 5))
        plt.plot(testo, label="real data")
        plt.plot(pred, label="predicted data")
        plt.legend()
        plt.show()


    ##卷积神经网络加lstm模型，deepsense模型
    def DEEPSENSE_RUN(self,traini, traino, testi, testo, epochs, batch, act="linear", dropout=0.1, loss="mae",
                      optimizer="nadam"):
        model = self.DEEPSENSE(traini)
        model.fit(traini, traino, epochs=epochs, batch_size=batch, verbose=2, shuffle=True)
        pred = model.predict(testi)
        print(metrics.r2_score(testo, pred))

        plt.figure(figsize=(20, 5))
        plt.plot(testo, label="real data")
        plt.plot(pred, label="predicted data")
        plt.legend()
        plt.show()
        MAE = mean_absolute_error(testo, pred)


    ##加了图像识别中的注意力模型的CNN+LSTM
    def DEEPSENSE2_RUN(self,traini, traino, testi, testo, epochs, batch, act="linear", dropout=0.1, loss="mae",
                       optimizer="nadam"):
        model = self.DEEPSENSE2(traini)
        model.fit(traini, traino, epochs=epochs, batch_size=batch, verbose=2, shuffle=True)
        pred = model.predict(testi)
        print(metrics.r2_score(testo, pred))

        plt.figure(figsize=(20, 5))
        plt.plot(testo, label="real data")
        plt.plot(pred, label="predicted data")
        plt.legend()
        plt.show()
        MAE = mean_absolute_error(testo, pred)


    ### define a lstm_model，简单的lstm模型，128个神经元的lstm加128个神经元的全连接
    def LSTM_model_stacking(self,inputs, activ_func="linear",
                            dropout=0.10, loss="mae", optimizer="adam"):
        model = Sequential()

        model.add(Reshape((inputs.shape[1], inputs.shape[2]), input_shape=(inputs.shape[1], inputs.shape[2], 1)))
        model.add(LSTM(64, input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=128))
        model.add(Dense(units=1))
        model.add(Activation(activ_func))

        model.compile(loss=loss, optimizer=optimizer)

        return model


    def stacking_pre(self,trainic,trainoc,batch = 10):
        epochs = 500

        lstm = self.LSTM_model_stacking(trainic)

        cnn_lstm = self.DEEPSENSE(trainic)

        cnn_am_lstm = self.DEEPSENSE2(trainic)

        lstm.fit(trainic, trainoc, epochs=epochs, batch_size=batch, verbose=2, shuffle=True)
        cnn_lstm.fit(trainic, trainoc, epochs=epochs, batch_size=batch, verbose=2, shuffle=True)
        cnn_am_lstm.fit(trainic, trainoc, epochs=epochs, batch_size=batch, verbose=2, shuffle=True)

        lstm.save_weights('weights/lstm.hdf5')
        cnn_lstm.save_weights('weights/cnn_lstm.hdf5')
        cnn_am_lstm.save_weights('weights/cnn_am_lstm.hdf5')

        lstm_model = self.LSTM_model(trainic)

        cnn_lstm_model = self.DEEPSENSE(trainic)

        cnn_am_lstm_model = self.DEEPSENSE2(trainic)

        lstm_model.load_weights('weights/lstm.hdf5')

        cnn_lstm_model.load_weights('weights/cnn_lstm.hdf5')

        cnn_am_lstm_model.load_weights('weights/cnn_am_lstm.hdf5')

        models = [lstm_model, cnn_lstm_model, cnn_am_lstm_model]

        return models


    def stacking(models, trainic):
        outputs = [model.outputs[0] for model in models]
        y = np.mean()[outputs]

        model = Model(inputs=trainic, outputs=y)

        return model




