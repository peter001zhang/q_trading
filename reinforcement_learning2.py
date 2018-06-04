import numpy as np
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers import Dropout
from keras import Model
from keras.layers import Input
from keras.layers import Multiply
from keras.layers import Add
from keras.layers import Subtract
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers import Lambda
from keras import backend as K
import os
import math
import data_processing
from keras.models import load_model
import pandas as pd
from random import choice

os.chdir(r'C:\cointree')
string=['aapl','amzn','blk','fb','googl','gs','jpm','msft','orcl','tsla']

class Reinforcement_learning2:
    ##read the data from the document,从文件中读取数据，返回data，string是股票的名称
    def read_data(self,string):
        data = pd.read_csv('%s.us.txt'% string)
        return data

    ##change the data formation into the matrix，将读取到的数据转化成神经网络能够理解的形式，并将时间和最后一行标签去掉，id是股票的名称
    def load_data(self,id):
        data = self.read_data(id)
        data = np.array(data)
        train_data = np.zeros((0,5))
        for row in data:
            cut_data = row[1:6]
            train_data = np.vstack((train_data,cut_data))
        return train_data


    ### seperate the data into training_input, output, testing_ input, output for lstm based on input data
    def seperate_RL_general(self,price, data, p, win):
        x = data
        y = price
        r = len(x) // 10
        c = 0
        train_input = np.zeros((r * p, win, x.shape[1]))
        for i in range(0, r * p):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    s = x[i + j][u]
                    train_input[i][j][u] = s

        test_input = np.zeros((len(x) - 2 * win - r * p, win, x.shape[1]))
        for i in range(r * p + win, len(x) - win):
            for j in range(0, win):
                for u in range(0, x.shape[1]):
                    s = x[i + j][u]
                    test_input[c][j][u] = s
            c = c + 1

        train_output = y[win - 1:r * p + win - 1]
        test_output = y[r * p + 2 * win - 1:]
        return train_input, test_input, train_output, test_output


    def RL_data(self,technical_up, technical_down, price_line, rsi, p, win):
        new_data = np.zeros((price_line.shape))
        for i in range(0, len(new_data)):
            if technical_up[i] - technical_down[i] != 0:
                new_data[i] = (price_line[i] - technical_down[i]) / (technical_up[i] - technical_down[i])
                new_data[i] = (new_data[i])
                if new_data[i] > 1:
                    new_data[i] = 1.1
                elif new_data[i] < 0:
                    new_data[i] = -0.1
            else:
                new_data[i] = 0.5
        rl_data = np.hstack((new_data, rsi))
        train_input, test_input, train_output, test_output = self.seperate_RL_general(new_data, rl_data, p, win)
        traini = np.reshape(train_input, (train_input.shape[0], train_input.shape[1], train_input.shape[2], -1))
        testi = np.reshape(test_input, (test_input.shape[0], test_input.shape[1], test_input.shape[2], -1))

        return traini, testi, train_output, test_output

    def test(self,separation, win,name):
        up = data_processing.VWAP_UP(name, 7, 4)
        down = data_processing.VWAP_DOWN(name, 7, 4)
        price = data_processing.close_price(name)
        rsi = data_processing.RSI_E(name, 14)
        traini, testi, traino, testo = self.RL_data(up, down, price, rsi,separation,win)
        return traini,testi,traino,testo


    def dqn_model(self,traini):  ##定义神经网络模型
        classes = ("buy", "short", "holdb", "holds", "sell", "cover")  ##动作的类别
        dropout = 0.1

        inputs = Input(shape=(traini.shape[1], traini.shape[2], traini.shape[3]))

        x = Conv2D(16, (2, traini.shape[2]), input_shape=(traini.shape[1], traini.shape[2], 1),
                   padding='valid', data_format='channels_last', activation='relu')(inputs)
        y1 = Conv2D(16, (2, traini.shape[2]), input_shape=(traini.shape[1], traini.shape[2], 1),
                    padding='valid', data_format='channels_last', activation='relu')(inputs)  ###采用图像识别的1*1卷积核连续卷积
        y2 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y1)  ###并与原来的数据进行点乘完成对重点
        y3 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y2)  ###特征的权重加大
        y = Conv2D(16, (1, 1), padding='valid', activation='relu')(y3)

        out = Multiply()([x, y])

        z1 = Reshape((traini.shape[1] - 1, 16), input_shape=(traini.shape[1] - 1, 1, 16))(out)
        z2 = Conv1D(8, 3, input_shape=(traini.shape[1] - 1, 16), activation='relu')(z1)  ####CNN提取重要特征
        z3 = Conv1D(4, 2, activation='relu')(z2)
        z4 = Dropout(dropout)(z3)
        z6 = Flatten()(z4)
        z7 = Dense(3, activation='linear')(z6)  ####输出三个值
        model = Model(inputs=inputs, outputs=z7)

        model.compile(loss='mae', optimizer='nadam')
        return model

    def mean(self,x):
        x1 = x - K.mean(x, axis=1, keepdims=True)
        return x1


    def dueling_model(self,traini):  ##定义神经网络模型
        classes = ("buy", "short", "holdb", "holds", "sell", "cover")  ##动作的类别
        dropout = 0.1

        inputs = Input(shape=(traini.shape[1], traini.shape[2], traini.shape[3]))

        x = Conv2D(16, (2, traini.shape[2]), input_shape=(traini.shape[1], traini.shape[2], 1),
                   padding='valid', data_format='channels_last', activation='relu')(inputs)
        y1 = Conv2D(16, (2, traini.shape[2]), input_shape=(traini.shape[1], traini.shape[2], 1),
                    padding='valid', data_format='channels_last', activation='relu')(inputs)  ###采用图像识别的1*1卷积核连续卷积
        y2 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y1)  ###并与原来的数据进行点乘完成对重点
        y3 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y2)  ###特征的权重加大
        y = Conv2D(16, (1, 1), padding='valid', activation='relu')(y3)

        out = Multiply()([x, y])

        z1 = Reshape((traini.shape[1] - 1, 16), input_shape=(traini.shape[1] - 1, 1, 16))(out)
        z2 = Conv1D(8, 3, input_shape=(traini.shape[1] - 1, 16), activation='relu')(z1)  ####CNN提取重要特征
        z3 = Conv1D(4, 2, activation='relu')(z2)
        z4 = Dropout(dropout)(z3)
        z6 = Flatten()(z4)

        a = Dense(1, activation='linear')(z6)
        a1 = RepeatVector(3)(a)
        a2 = Reshape((3,), input_shape=(3, 1))(a1)

        b = Dense(3, activation='linear')(z6)
        b1 = Lambda(self.mean, output_shape=(3,))(b)
        c2 = Add()([a2, b1])  ####输出三个值
        model = Model(inputs=inputs, outputs=c2)
        model.compile(loss='mae', optimizer='nadam')
        return model


    def action(self,state, model, e):
        action = [0, 1, 2]  ###对应买入，卖出，持有  或者卖空，平仓，持有
        if np.random.rand() <= e:  # 若0-1的随机数大于贪婪值，则根据可选的动作随机输出一个
            act = choice(action)
            return act
        else:
            act = np.argmax(model.predict(state))
            return act


    def DQN_long(self,name, model_name):
        e = 1.0
        e_decay = 0.995
        e_min = 0.01
        n_train = 50000
        l = 0.99
        win = 10
        start_point = 0  ####用于在replay中删除过于久远的数据

        traini, testi, traino, testo = self.test(9,win,name)

        qmemory = []  ##记忆输入（环境值）
        q_memory = []  ##下一个记忆的输入值（环境值）
        reward_memory = []  ##奖励记忆
        reward_buffer = np.zeros(3, )
        done = []
        buffer = 1  ##用来记录上一次买入或者卖出操作
        buffer_price = traino[0]  ###用来记录上一次买入或者卖空时的价格（不是真正的价格，是映射到vwap上的价格）
        q_replay = np.zeros((1, 3))  ## 记忆q值
        q_max = np.zeros((1, 3))

        model_action = self.dqn_model(traini)
        model_replay = self.dqn_model(traini)

        for j in range(1, n_train):
            i = j % (len(traini) - 1)  ##防止迭代次数超过数据量
            e = e * e_decay  ###将贪婪值随迭代次数减小
            e = max(e, e_min)  ###最低不能低于e_min
            print(reward_buffer)
            state = np.reshape(traini[i], (1, win, traini.shape[2], 1))  ##获取这时刻的环境输入值
            state_ = np.reshape(traini[i + 1], (1, win, traini.shape[2], 1))  ##获取这时刻的环境输入值
            print(state.shape)
            act = self.action(state, model_replay, e)
            done_buffer = 0  ###结束符号
            if act == 0:  ####定义奖励，更改价格坐标，更改结束符号
                if buffer == 1:
                    buffer = 0
                    reward_buffer[0] = buffer_price - traino[i] - 0.1
                    reward_buffer[1] = -0.1
                    reward_buffer[2] = 0
                    buffer_price = traino[i]
                    done_buffer = 1
                else:
                    print("wrong action")
                    reward_buffer[0] = -1
                    reward_buffer[1] = traino[i] - buffer_price - 0.1
                    reward_buffer[2] = 0
            elif act == 1:
                if buffer == 1:
                    print("wrong action")
                    reward_buffer[0] = buffer_price - traino[i] - 0.1
                    reward_buffer[1] = -0.1
                    reward_buffer[2] = 0
                else:
                    buffer = 1
                    reward_buffer[0] = -0.1
                    reward_buffer[1] = traino[i] - buffer_price - 0.1
                    reward_buffer[2] = 0
                    buffer_price = traino[i]
                    done_buffer = 1
            else:
                if buffer == 1:
                    reward_buffer[0] = buffer_price - traino[i] - 0.1
                    reward_buffer[1] = -0.1
                    reward_buffer[2] = 0
                else:
                    buffer = 1
                    reward_buffer[0] = -0.1
                    reward_buffer[1] = traino[i] - buffer_price - 0.1
                    reward_buffer[2] = 0

            done.append(done_buffer)
            qmemory.append(state)
            q_memory.append(state_)
            reward_memory.append(reward_buffer)

            if j % 50 == 0:  ##每50次尝试之后对记忆中的数据进行训练

                for times in range(0, 500):
                    k = choice(range(start_point, len(qmemory)))  ###将记忆中的数据再处理，使用新的权重得到新的目标q值
                    q = q_memory[k]
                    r_replay1 = reward_memory[k][0]
                    r_replay2 = reward_memory[k][1]
                    r_replay3 = reward_memory[k][2]

                    q_replay[0][0] = r_replay1 + l * model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                    q_replay[0][1] = r_replay2 + l * model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                    q_replay[0][2] = r_replay3 + l * model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                    m = qmemory[k]  ##这一时刻的输入
                    model_replay.fit(m, q_replay, epochs=1, batch_size=1, verbose=2)  ###训练记忆模型
            if j % 500 == 0:
                weights = model_replay.get_weights()
                action_weights = model_action.get_weights()  ###将记忆模型的权重送给动作选择模型
                for i in range(0, len(action_weights)):
                    action_weights[i] = weights[i]
                model_action.set_weights(action_weights)

                start_point = start_point + 20

            q_max[0][0] = reward_buffer[0] + l * model_replay.predict(state_)[0][np.argmax(model_action.predict(state_))]
            q_max[0][1] = reward_buffer[1] + l * model_replay.predict(state_)[0][np.argmax(model_action.predict(state_))]
            q_max[0][2] = reward_buffer[2] + l * model_replay.predict(state_)[0][np.argmax(model_action.predict(state_))]
            print("this is", j, "times epochs")
            model_replay.fit(state, q_max, epochs=1, batch_size=1, verbose=2)  ##训练模型
            if j % 999 == 0:
                model_replay.save(model_name)


    def DQN_short(self,name, model_name):
        e = 1.0
        e_decay = 0.995
        e_min = 0.01
        n_train = 50000
        l = 0.99
        win = 10
        start_point = 0  ####用于在replay中删除过于久远的数据

        traini, testi, traino, testo = self.test(9,win,name)

        qmemory = []  ##记忆输入（环境值）
        q_memory = []  ##下一个记忆的输入值（环境值）
        reward_memory = []  ##奖励记忆
        reward_buffer = np.zeros(3, )
        done = []
        action_memory = []
        buffer = 1  ##用来记录上一次买入或者卖出操作
        buffer_price = traino[0]  ###用来记录上一次买入或者卖空时的价格（不是真正的价格，是映射到vwap上的价格）
        q_replay = np.zeros((1, 3))  ## 记忆q值
        q_max = np.zeros((1, 3))

        model_action = self.dqn_model(traini)
        model_replay = self.dqn_model(traini)

        for j in range(1, n_train):
            i = j % (len(traini) - 1)  ##防止迭代次数超过数据量
            e = e * e_decay  ###将贪婪值随迭代次数减小
            e = max(e, e_min)  ###最低不能低于e_min
            print(reward_buffer)
            state = np.reshape(traini[i], (1, win, traini.shape[2], 1))  ##获取这时刻的环境输入值
            state_ = np.reshape(traini[i + 1], (1, win, traini.shape[2], 1))  ##获取这时刻的环境输入值
            print(state.shape)
            act = self.action(state, model_replay, e)
            done_buffer = 0  ###结束符号
            if act == 0:  ####定义奖励，更改价格坐标，更改结束符号
                if buffer == 1:
                    buffer = 0
                    reward_buffer[0] = -buffer_price + traino[i] - 0.1
                    reward_buffer[1] = -0.1
                    reward_buffer[2] = 0
                    buffer_price = traino[i]
                    done_buffer = 1
                else:
                    print("wrong action")
                    reward_buffer[0] = -1
                    reward_buffer[1] = -traino[i] + buffer_price - 0.1
                    reward_buffer[2] = 0
            elif act == 1:
                if buffer == 1:
                    print("wrong action")
                    reward_buffer[0] = -buffer_price + traino[i] - 0.1
                    reward_buffer[1] = -0.1
                    reward_buffer[2] = 0
                else:
                    buffer = 1
                    reward_buffer[0] = -0.1
                    reward_buffer[1] = -traino[i] + buffer_price - 0.1
                    reward_buffer[2] = 0
                    buffer_price = traino[i]
                    done_buffer = 1
            else:
                if buffer == 1:
                    reward_buffer[0] = -buffer_price + traino[i] - 0.1
                    reward_buffer[1] = -0.1
                    reward_buffer[2] = 0
                else:
                    buffer = 1
                    reward_buffer[0] = -0.1
                    reward_buffer[1] = -traino[i] + buffer_price - 0.1
                    reward_buffer[2] = 0

            done.append(done_buffer)
            qmemory.append(state)
            q_memory.append(state_)
            reward_memory.append(reward_buffer)
            action_memory.append(act)

            if j % 50 == 0:  ##每50次尝试之后对记忆中的数据进行训练

                for times in range(0, 500):
                    k = choice(range(start_point, len(qmemory)))  ###将记忆中的数据再处理，使用新的权重得到新的目标q值
                    q = q_memory[k]
                    r_replay1 = reward_memory[k][0]
                    r_replay2 = reward_memory[k][1]
                    r_replay3 = reward_memory[k][2]
                    done_replay = done[k]
                    action_replay = action_memory[k]
                    if done_replay == 1:
                        q_replay[0][action_replay] = reward_memory[k][action_replay]
                        q_replay[0][(action_replay + 1) % 3] = reward_memory[k][(action_replay + 1) % 3] + l * \
                                                               model_replay.predict(q)[0][
                                                                   np.argmax(model_action.predict(q))]
                        q_replay[0][(action_replay + 2) % 3] = reward_memory[k][(action_replay + 2) % 3] + l * \
                                                               model_replay.predict(q)[0][
                                                                   np.argmax(model_action.predict(q))]
                    else:
                        q_replay[0][0] = r_replay1 + l * model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                        q_replay[0][1] = r_replay2 + l * model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                        q_replay[0][2] = r_replay3 + l * model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                    m = qmemory[k]  ##这一时刻的输入
                    model_replay.fit(m, q_replay, epochs=1, batch_size=1, verbose=2)  ###训练记忆模型
            if j % 500 == 0:
                weights = model_replay.get_weights()
                action_weights = model_action.get_weights()  ###将记忆模型的权重送给动作选择模型
                for i in range(0, len(action_weights)):
                    action_weights[i] = weights[i]
                model_action.set_weights(action_weights)

                start_point = start_point + 20

            if done_buffer == 0:
                q_max[0][0] = reward_buffer[0] + l * model_replay.predict(state_)[0][
                    np.argmax(model_action.predict(state_))]
                q_max[0][1] = reward_buffer[1] + l * model_replay.predict(state_)[0][
                    np.argmax(model_action.predict(state_))]
                q_max[0][2] = reward_buffer[2] + l * model_replay.predict(state_)[0][
                    np.argmax(model_action.predict(state_))]
            else:
                q_max[0][act] = reward_buffer[act]
                q_max[0][(act + 1) % 3] = reward_buffer[(act + 1) % 3] + l * model_replay.predict(state_)[0][
                    np.argmax(model_action.predict(state_))]
                q_max[0][(act + 2) % 3] = reward_buffer[(act + 2) % 3] + l * model_replay.predict(state_)[0][
                    np.argmax(model_action.predict(state_))]
            print("this is", j, "times epochs")
            model_replay.fit(state, q_max, epochs=1, batch_size=1, verbose=2)  ##训练模型
            if j % 999 == 0:
                model_replay.save(model_name)


    ##利用做多模型和做空模型对输出动作进行选择
    def policy(self,name):
        win = 10
        long_model = load_model('dqn_long.h5')
        short_model = load_model('dqn_short.h5')

        traini, testi, traino, testo =self.test(9,win,name)
        long_buffer = 1
        short_buffer = 1
        long_short_buffer = 0  ###定义现在是在short的状态中还是在buy的状态中
        act_storage = []
        for i in range(0, len(testi)):

            state = testi[i]
            act_long = self.action(state, long_model, 0.01)
            act_short = self.action(state, short_model, 0.01)

            ##如果做多模型的上一次记录是buy，则只能输出做多模型的动作，如果做多模型的上一次记录是sell，则有极大可能做空模型会输出short指令，则我们判断
            ##如果做空模型的输出是持仓，则我们输出做多模型的动作（多数情况下也是持仓，没有影响），如果做空模型输出的动作是short或者cover，则我们根据做空
            ##模型上一次的记录做出判断，动作是否有效，这样就达到了使用两个模型进行互相干预输出动作的目的。
            ##在这个模型中，做多模型的优先级明显高于做空模型，在未来的设计中，我会尝试加入预测的趋势值，判断做多和做空模型哪一个的优先级更高
            if long_buffer == 0:
                act = act_long
                act_storage.append(act)
                if act == 0:
                    print("wrong action")
                    print("hold")
                    act = 2
                elif act == 1:
                    print("sell the portfolio")
                    long_buffer = 1
                else:
                    print("hold the position")
            else:
                if act_short == 2:
                    act = act_long
                    if act == 1:
                        print("wrong action")
                        print("hold")
                        act = 2
                    elif act == 0:
                        print("buy the portfolio")
                        long_buffer = 0
                    else:
                        print("hold the position")
                else:
                    act = act_short
                    if short_buffer == 0:
                        if act == 0:
                            print("wrong action")
                            print("hold")
                            act = 2
                        else:
                            print("cover the portfolio")
                            short_buffer = 0
                    if short_buffer == 1:
                        if act == 1:
                            print("wrong action")
                            print("hold")
                            act = 2
                        else:
                            print("short the portfolio")
                            short_buffer = 0
            return act





