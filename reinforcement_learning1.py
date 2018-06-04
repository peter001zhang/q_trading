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
import data_processing
import math
from keras.models import load_model
import pandas as pd
from random import choice

os.chdir(r'C:\cointree')
string=['aapl','amzn','blk','fb','googl','gs','jpm','msft','orcl','tsla']

class Reinforcement_learning1:
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




    def seperate_dqn_normal(self,name, p, win):  ####对数据进行归一化，并处理成神经网络能够理解的数据结构
        x = self.load_data(name)
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
        traini = np.reshape(train_input, (train_input.shape[0], train_input.shape[1], train_input.shape[2], -1))
        testi = np.reshape(test_input, (test_input.shape[0], test_input.shape[1], test_input.shape[2], -1))
        return traini, testi, train_output, test_output


    def dqn_model(self,traini1):  ##定义神经网络模型
        classes = ("buy", "short", "holdb", "holds", "sell", "cover")  ##动作的类别
        dropout = 0.1

        inputs = Input(shape=(traini1.shape[1], traini1.shape[2] + 6, traini1.shape[3]))

        x = Conv2D(16, (2, traini1.shape[2] + 6), input_shape=(traini1.shape[1], traini1.shape[2] + 6, 1),
                   padding='valid', data_format='channels_last', activation='linear')(inputs)
        y1 = Conv2D(16, (2, traini1.shape[2] + 6), input_shape=(traini1.shape[1], traini1.shape[2] + 6, 1),
                    padding='valid', data_format='channels_last', activation='linear')(inputs)  ###采用图像识别的1*1卷积核连续卷积
        y2 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y1)  ###并与原来的数据进行点乘完成对重点
        y3 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y2)  ###特征的权重加大
        y = Conv2D(16, (1, 1), padding='valid', activation='relu')(y3)

        out = Multiply()([x, y])

        z1 = Reshape((traini1.shape[1] - 1, 16), input_shape=(traini1.shape[1] - 1, 1, 16))(out)
        z2 = Conv1D(8, 3, input_shape=(traini1.shape[1] - 1, 16), activation='linear')(z1)  ####CNN提取重要特征
        z3 = Conv1D(4, 2, activation='linear')(z2)
        z4 = Dropout(dropout)(z3)
        z6 = Flatten()(z4)
        z7 = Dense(2, activation='linear')(z6)  ####输出两个值
        model = Model(inputs=inputs, outputs=z7)

        model.compile(loss='mae', optimizer='nadam')
        return model


    def dueling_model(self,traini1):  ##定义神经网络模型
        classes = ("buy", "short", "holdb", "holds", "sell", "cover")  ##动作的类别
        dropout = 0.1

        inputs = Input(shape=(traini1.shape[1], traini1.shape[2] + 6, traini1.shape[3]))

        x = Conv2D(16, (2, traini1.shape[2] + 6), input_shape=(traini1.shape[1], traini1.shape[2] + 6, 1),
                   padding='valid', data_format='channels_last', activation='linear')(inputs)
        y1 = Conv2D(16, (2, traini1.shape[2] + 6), input_shape=(traini1.shape[1], traini1.shape[2] + 6, 1),
                    padding='valid', data_format='channels_last', activation='linear')(inputs)  ###采用图像识别的1*1卷积核连续卷积
        y2 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y1)  ###并与原来的数据进行点乘完成对重点
        y3 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y2)  ###特征的权重加大
        y = Conv2D(16, (1, 1), padding='valid', activation='relu')(y3)

        out = Multiply()([x, y])

        z1 = Reshape((traini1.shape[1] - 1, 16), input_shape=(traini1.shape[1] - 1, 1, 16))(out)
        z2 = Conv1D(8, 3, input_shape=(traini1.shape[1] - 1, 16), activation='linear')(z1)  ####CNN提取重要特征
        z3 = Conv1D(4, 2, activation='linear')(z2)
        z4 = Dropout(dropout)(z3)
        #     z5 = LSTM(128)(z4)                                                                           ###LSTM 对连续的时间序列进行处理
        #     z6 = Dropout(dropout)(z5)
        z6 = Flatten()(z4)

        a = Dense(1, activation='linear')(z6)
        a1 = RepeatVector(2)(a)
        a2 = Reshape((2,), input_shape=(2, 1))(a1)

        b = Dense(2, activation='linear')(z6)
        b1 = Lambda(self.mean, output_shape=(2,))(b)
        c2 = Add()([a2, b1])  ####输出两个值
        model = Model(inputs=inputs, outputs=c2)
        model.compile(loss='mae', optimizer='nadam')
        return model

    def mean(self,x):
        x1 = x - K.mean(x, axis=1, keepdims=True)
        return x1


    ##动作选择
    def action(self,data, act_last, model_replay, i, e):
        string1 = [2, 4]  ##三种不同的动作选择限制
        string2 = [3, 5]
        string3 = [0, 1]
        action = [0, 1]
        if np.random.rand() >= e:  # 若0-1的随机数大于贪婪值，则根据可选的动作随机输出一个
            if act_last == 0:
                act = choice(action)
                act_name = string1[act]
                print(act_name)
                return act_name, act
            elif act_last == 1:
                act = choice(action)
                act_name = string2[act]
                print(act_name)
                return act_name, act
            elif act_last == 2:
                act = choice(action)
                act_name = string1[act]
                print(act_name)
                return act_name, act
            elif act_last == 3:
                act = choice(action)
                act_name = string2[act]
                print(act_name)
                return act_name, act
            elif act_last == 4:
                act = choice(action)
                act_name = string3[act]
                print(act_name)
                return act_name, act
            else:
                act = choice(action)
                act_name = string3[act]
                print(act_name)
                return act_name, act
        else:  ####若随机数小于贪婪值 则根据训练好的网络模型选择q值最大的进行输出
            if act_last == 0:
                p1 = model_replay.predict(data)
                p = p1[0]
                p = list(p)
                act = p.index(max(p))
                act_name = string1[act]
                print(act_name)
                return act_name, act
            elif act_last == 1:
                p1 = model_replay.predict(data)
                p = p1[0]
                p = list(p)
                act = p.index(max(p))
                act_name = string2[act]
                print(act_name)
                return act_name, act
            elif act_last == 2:
                p1 = model_replay.predict(data)
                p = p1[0]
                p = list(p)
                act = p.index(max(p))
                act_name = string1[act]
                print(act_name)
                return act_name, act
            elif act_last == 3:
                p1 = model_replay.predict(data)
                p = p1[0]
                p = list(p)
                act = p.index(max(p))
                act_name = string2[act]
                print(act_name)
                return act_name, act
            elif act_last == 4:
                p1 = model_replay.predict(data)
                p = p1[0]
                p = list(p)
                act = p.index(max(p))
                act_name = string3[act]
                print(act_name)
                return act_name, act
            else:
                p1 = model_replay.predict(data)
                p = p1[0]
                p = list(p)
                act = p.index(max(p))
                act_name = string3[act]
                print(act_name)
                return act_name, act


    def reward(self,a, c, d, price, win, x, i):  ##采用价格收益的模式对奖励进行设计
        if a[d] == 0:
            r0 = price[win + i] - price[win - 1 + i]  ##持有没有奖励也没有惩罚
            r1 = price[win - 1 + i] - x  ##价格差作为奖励

            ##若选择的动作为sell，则上一步价格清零
            if a[c] == 4:
                x = 0
            else:
                x = x
            return r0, r1, x
        ## 根据上一步的动作（short），选择这一步能有选择的动作
        elif a[d] == 1:
            r0 = price[win - 1 + i] - price[win + i]
            r1 = x - price[win - 1 + i]
            ##若选择的动作为cover，则上一步价格清零
            if a[c] == 5:
                x = 0
            else:
                x = x
            return r0, r1, x

        ## 根据上一步的动作（holdb（做多时持有）），选择这一步能有选择的动作
        elif a[d] == 2:
            r0 = price[win + i] - price[win - 1 + i]
            r1 = price[win - 1 + i] - x

            ##若选择的动作为sell，则上一步价格清零
            if a[c] == 4:
                x = 0
            else:
                x = x
            return r0, r1, x


        ## 根据上一步的动作（holds（做空时持有）），选择这一步能有选择的动作
        elif a[d] == 3:
            r0 = price[win - 1 + i] - price[win + i]
            r1 = x - price[win - 1 + i]

            ##若选择的动作为cover，则上一步价格清零
            if a[c] == 5:
                x = 0
            else:
                x = x
            return r0, r1, x


        ## 根据上一步的动作（sell），选择这一步能有选择的动作
        elif a[d] == 4:
            r0 = price[win - 2 + i] - price[win - 1 + i]
            r1 = price[win - 1 + i] - price[win - 2 + i]
            x = price[win - 1 + i]  ##此时的价格计入x
            return r0, r1, x

        ## 根据上一步的动作（cover），选择这一步能有选择的动作
        else:
            r0 = price[win - 2 + i] - price[win - 1 + i]
            r1 = price[win - 1 + i] - price[win - 2 + i]
            x = price[win - 1 + i]
            return r0, r1, x


    def value(self,a, c, d, price, win, x, i):  ##输出收益价值
        if a[d] == 0:
            r0 = 0  ##持有没有奖励也没有惩罚
            r1 = price[win - 1 + i] - x  ##价格差作为奖励

            ##若选择的动作为sell，则上一步价格清零
            if a[c] == 4:
                x = 0
            else:
                x = x
            return r0, r1, x
        ## 根据上一步的动作（short），选择这一步能有选择的动作
        elif a[d] == 1:
            r0 = 0
            r1 = x - price[win - 1 + i]
            ##若选择的动作为cover，则上一步价格清零
            if a[c] == 5:
                x = 0
            else:
                x = x
            return r0, r1, x

        ## 根据上一步的动作（holdb（做多时持有）），选择这一步能有选择的动作
        elif a[d] == 2:
            r0 = 0
            r1 = price[win - 1 + i] - x

            ##若选择的动作为sell，则上一步价格清零
            if a[c] == 4:
                x = 0
            else:
                x = x
            return r0, r1, x


        ## 根据上一步的动作（holds（做空时持有）），选择这一步能有选择的动作
        elif a[d] == 3:
            r0 = 0
            r1 = x - price[win - 1 + i]

            ##若选择的动作为cover，则上一步价格清零
            if a[c] == 5:
                x = 0
            else:
                x = x
            return r0, r1, x


        ## 根据上一步的动作（sell），选择这一步能有选择的动作
        elif a[d] == 4:
            r0 = -0.1
            r1 = -0.1
            x = price[win - 1 + i]  ##此时的价格计入x
            return r0, r1, x

        ## 根据上一步的动作（cover），选择这一步能有选择的动作
        else:
            r0 = -0.1
            r1 = -0.1
            x = price[win - 1 + i]
            return r0, r1, x


    def reward2(self,a, c, d, price, win, x, i):  ##采用固定reward的格式设计奖励
        if a[d] == 0:
            if price[win + i] >= price[win - 1 + i]:
                r0 = 1
            else:
                r0 = -1
            if price[win + i - 1] >= price[win + i - 2]:
                r1 = 1
            else:
                r1 = -1

            ##若选择的动作为sell，则上一步价格清零
            if a[c] == 4:
                x = 0
            else:
                x = x
            return r0, r1, x
        ## 根据上一步的动作（short），选择这一步能有选择的动作
        elif a[d] == 1:
            if price[win + i] <= price[win - 1 + i]:
                r0 = 1
            else:
                r0 = -1
            if price[win + i - 1] <= price[win + i - 2]:
                r1 = 1
            else:
                r1 = -1
            if a[c] == 5:
                x = 0
            else:
                x = x
            return r0, r1, x

        ## 根据上一步的动作（holdb（做多时持有）），选择这一步能有选择的动作
        elif a[d] == 2:
            if price[win + i] >= price[win - 1 + i]:
                r0 = 1
            else:
                r0 = -1
            if price[win + i - 1] >= price[win + i - 2]:
                r1 = 1
            else:
                r1 = -1
            ##若选择的动作为sell，则上一步价格清零
            if a[c] == 4:
                x = 0
            else:
                x = x
            return r0, r1, x


        ## 根据上一步的动作（holds（做空时持有）），选择这一步能有选择的动作
        elif a[d] == 3:
            if price[win + i] <= price[win - 1 + i]:
                r0 = 1
            else:
                r0 = -1
            if price[win + i - 1] <= price[win + i - 2]:
                r1 = 1
            else:
                r1 = -1

            ##若选择的动作为cover，则上一步价格清零
            if a[c] == 5:
                x = 0
            else:
                x = x
            return r0, r1, x


        ## 根据上一步的动作（sell），选择这一步能有选择的动作
        elif a[d] == 4:
            if price[win + i] >= price[win - 1 + i]:
                r0 = 1
                r1 = -1
            else:
                r0 = -1
                r1 = 1

            x = price[win - 1 + i]  ##此时的价格计入x
            return r0, r1, x

        ## 根据上一步的动作（cover），选择这一步能有选择的动作
        else:
            if price[win + i] >= price[win - 1 + i]:
                r0 = 1
                r1 = -1
            else:
                r0 = -1
                r1 = 1
            x = price[win - 1 + i]
            return r0, r1, x

    def add_data(self,win):    ###用于将上一个动作对应的值添加到训练集中，对应数据为one——hot格式
        t_add = np.zeros((6,1,win,6,1))
        t_buy = np.zeros((1,win,6,1))
        t_short = np.zeros((1,win,6,1))
        t_holdb = np.zeros((1,win,6,1))
        t_holds = np.zeros((1,win,6,1))
        t_sell = np.zeros((1,win,6,1))
        t_cover = np.zeros((1,win,6,1))
        for i in range(0,1):
            for j in range(0,win):
                t_buy[i][j][0] = 1
                t_short[i][j][1] = 1
                t_holdb[i][j][2] = 1
                t_holds[i][j][3] = 1
                t_sell[i][j][4] = 1
                t_cover[i][j][5] = 1
        t_add[0] = t_buy
        t_add[1] = t_short
        t_add[2] = t_holdb
        t_add[3] = t_holds
        t_add[4] = t_sell
        t_add[5] = t_cover
        add_data = t_add
        return add_data


    def DQN(self,name, model_name='dqn.h5'):
        traini, testi, traino, testo = self.seperate_dqn_normal(name, 9, 10)
        t = traini  ##训练集

        n_train = 50000  ###训练次数
        l = 0.99  ##未来奖励率
        win = 10  ##数据队列长度
        x = 0  ##保存上一步价格用的空数据结构
        x_test = 0
        total_value = 0
        a = np.zeros(2, )  ##最近动作队列，两个值的队列
        q_max = np.zeros((1, 2))  ##q值拟合目标值， 最大奖励值
        memory = np.zeros((1, win, traini.shape[2] + 6, 1))  ##记忆输入（环境值）
        q_memory = np.zeros((1, win, traini.shape[2] + 6, 1))  ##下一个记忆的输入值（环境值）
        reward_memory = np.zeros((1, 2, 1))  ##奖励记忆
        reward_buffer = np.zeros((2, 1))  ##用于处理奖励值的buffer
        memory = np.delete(memory, 0, axis=0)
        q_memory = np.delete(q_memory, 0, axis=0)  ##删除第一条全零数据
        reward_memory = np.delete(reward_memory, 0, axis=0)
        action_memory = []  ###用于记忆动作
        total_value_memory = []
        a[1] = 4  ###定义第零条数据之前的动作为sell
        e = 0.5  ##贪婪率， 初始0.5代表开始全靠猜
        price = data_processing.close_price(name)  ##价格
        start_point = 0
        model_action = self.dqn_model(traini)  ##实现动作选择模型，又可以理解为数据获取模型
        model_replay = self.dqn_model(traini)  ##记忆训练模型， replay 模型
        print(model_action.summary())
        ##主循环
        for j in range(1, n_train):

            print("the", j, "time run, start:")  ##第一次迭代开始
            i = j % (len(t) - 1)  ##防止迭代次数超过数据量
            c = (j + 1) % 2  ## 这一步在a中的位置
            d = (j) % 2  ##上一步在a中的位置

            s = 0.01 * (j // 1000)  ##随训练次数的增加而增加
            s = s + e
            state = np.reshape(t[i], (1, win, traini.shape[2], 1))  ##获取这时刻的环境输入值
            data = self.add_data(win)  ##获取用于添加到训练集中的动作数据
            state = np.dstack((state, data[a[d]]))  ##将上一次动作对应的数据添加到训练集输入中
            act, act_number = self.action(state, a[d], model_replay, j, s)  ###获取本次动作信息
            print(state.shape)
            a[c] = act  ##将这次动作放到动作队列中
            r0, r1, x = self.reward2(a, c, d, price, win, x, i)  ###获取对应的奖励值，更新价格参数
            print(x)
            r0_test, r1_test, x_test = self.value(a, c, d, price, win, x_test, i)
            if act_number == 0:
                total_value = total_value + r0_test
            else:
                total_value = total_value + r1_test

            fit_i = np.reshape(t[i], (1, win, traini.shape[2], 1))  ##变换输入格式，从3维度变换为4维度
            fit_i = np.dstack((fit_i, data[a[d]]))  ##将上一步动作值加入此时刻的环境变量中
            fit_o = np.reshape(t[i + 1], (1, win, traini.shape[2], 1))
            fit_o = np.dstack((fit_o, data[a[c]]))  ##将这一步的动作值加入到下一时刻的环境变量中
            total_value_memory.append(total_value)
            if total_value <= 30:
                if act_number == 0:
                    q_max[0][0] = -20
                    q_max[0][1] = r1
                    total_value = 0
                else:
                    q_max[0][0] = r0
                    q_max[0][1] = -20
                    total_value = 0
            elif total_value >= 40:
                if act_number == 0:
                    q_max[0][0] = 20
                    q_max[0][1] = r1
                    total_value = 0
                else:
                    q_max[0][0] = r0
                    q_max[0][1] = 20
                    total_value = 0
            else:
                q_max[0][0] = r0 + l * np.max(model_action.predict(fit_o))  ##q值的目标值，或者可以称为label
                q_max[0][1] = r1 + l * np.max(model_action.predict(fit_o))

            memory = np.vstack((memory, fit_i))  ##将此时刻处理过的环境值加入到环境记忆中
            q_memory = np.vstack((q_memory, fit_o))  ##将此时刻的q最大值加入到label记忆中
            action_memory.append(act_number)  ##将动作添加到动作记忆中
            total_value_memory.append(total_value)
            if total_value <= 30:
                if act_number == 0:
                    reward_buffer[0][0] = -20
                    reward_buffer[1][0] = r1
                else:
                    reward_buffer[0][0] = r0
                    reward_buffer[1][0] = -20
            elif total_value >= 40:
                if act_number == 0:
                    reward_buffer[0][0] = 20
                    reward_buffer[1][0] = r1
                else:
                    reward_buffer[0][0] = r0
                    reward_buffer[1][0] = 20
            else:
                reward_buffer[0][0] = r0
                reward_buffer[1][0] = r1
            r = np.reshape(reward_buffer, (1, 2, 1))
            reward_memory = np.vstack((reward_memory, r))  ##添加本次奖励到记忆中

            if j % 50 == 0:  ##每50次尝试之后对记忆中的数据进行训练
                q_replay = np.zeros((1, 2))  ## 记忆q值
                for times in range(0, 500):
                    k = choice(range(start_point, len(q_memory)))  ###将记忆中的数据再处理，使用新的权重得到新的目标q值
                    q = np.reshape(q_memory[k], (1, win, traini.shape[2] + 6, 1))
                    r2 = reward_memory[k][0]
                    r3 = reward_memory[k][1]
                    act_replay = action_memory[k]
                    if total_value_memory[k] <= 30 or total_value_memory[k] >= 40:
                        q_replay[0][act_replay] = reward_memory[k][act_replay]
                        q_replay[0][(act_replay + 1) % 2] = reward_memory[k][(act_replay + 1) % 2] + np.max(
                            model_action.predict(q))
                    else:
                        q_replay[0][0] = r2 + l * np.max(model_action.predict(q))
                        q_replay[0][1] = r3 + l * np.max(model_action.predict(q))
                    m = memory[k]  ##这一时刻的输入
                    m = np.reshape(m, (1, win, traini.shape[2] + 6, 1))
                    model_replay.fit(m, q_replay, epochs=1, batch_size=1, verbose=2)  ###训练记忆模型
            if j % 500 == 0:
                weights = model_replay.get_weights()
                action_weights = model_action.get_weights()  ###将记忆模型的权重送给动作选择模型
                for i in range(0, len(action_weights)):
                    action_weights[i] = weights[i]
                model_action.set_weights(action_weights)
                start_point = start_point + 50

            model_replay.fit(fit_i, q_max, epochs=1, batch_size=1, verbose=2)  ##训练模型
            if j % 999 == 0:
                model_replay.save(model_name)


    def DDQN_DUELING(self,name, select_number, model_name='dqn_dueling.h5'):
        traini, testi, traino, testo = self.seperate_dqn_normal(name, 9, 10)
        t = traini  ##训练集

        n_train = 50000  ###训练次数
        l = 0.99  ##未来奖励率
        win = 10  ##数据队列长度
        x = 0  ##保存上一步价格用的空数据结构
        x_test = 0
        total_value = 0
        a = np.zeros(2, )  ##最近动作队列，两个值的队列
        q_max = np.zeros((1, 2))  ##q值拟合目标值， 最大奖励值
        memory = np.zeros((1, win, traini.shape[2] + 6, 1))  ##记忆输入（环境值）
        q_memory = np.zeros((1, win, traini.shape[2] + 6, 1))  ##下一个记忆的输入值（环境值）
        reward_memory = np.zeros((1, 2, 1))  ##奖励记忆
        reward_buffer = np.zeros((2, 1))  ##用于处理奖励值的buffer
        memory = np.delete(memory, 0, axis=0)
        q_memory = np.delete(q_memory, 0, axis=0)  ##删除第一条全零数据
        reward_memory = np.delete(reward_memory, 0, axis=0)
        action_memory = []  ###用于记忆动作
        total_value_memory = []
        a[1] = 4  ###定义第零条数据之前的动作为sell
        e = 0.5  ##贪婪率， 初始0.5代表开始全靠猜
        price = data_processing.close_price(name)  ##价格
        start_point = 0
        if select_number == 0:
            model_action = self.dqn_model(traini)  ##实现动作选择模型，又可以理解为数据获取模型
            model_replay = self.dqn_model(traini)  ##记忆训练模型， replay 模型
        else:
            model_action = self.dueling_model(traini)  ##实现动作选择模型，又可以理解为数据获取模型
            model_replay = self.dueling_model(traini)  ##记忆训练模型， replay 模型
        print(model_action.summary())
        ##主循环
        for j in range(1, n_train):

            print("the", j, "time run, start:")  ##第一次迭代开始
            i = j % (len(t) - 1)  ##防止迭代次数超过数据量
            c = (j + 1) % 2  ## 这一步在a中的位置
            d = (j) % 2  ##上一步在a中的位置

            s = 0.01 * (j // 1000)  ##随训练次数的增加而增加
            s = s + e
            state = np.reshape(t[i], (1, win, traini.shape[2], 1))  ##获取这时刻的环境输入值
            data = self.add_data(win)  ##获取用于添加到训练集中的动作数据
            state = np.dstack((state, data[a[d]]))  ##将上一次动作对应的数据添加到训练集输入中
            act, act_number = self.action(state, a[d], model_replay, j, s)  ###获取本次动作信息
            print(state.shape)
            a[c] = act  ##将这次动作放到动作队列中
            r0, r1, x = self.reward2(a, c, d, price, win, x, i)  ###获取对应的奖励值，更新价格参数
            print(x)
            r0_test, r1_test, x_test = self.value(a, c, d, price, win, x_test, i)
            if act_number == 0:
                total_value = total_value + r0_test
            else:
                total_value = total_value + r1_test

            fit_i = np.reshape(t[i], (1, win, traini.shape[2], 1))  ##变换输入格式，从3维度变换为4维度
            fit_i = np.dstack((fit_i, data[a[d]]))  ##将上一步动作值加入此时刻的环境变量中
            fit_o = np.reshape(t[i + 1], (1, win, traini.shape[2], 1))
            fit_o = np.dstack((fit_o, data[a[c]]))  ##将这一步的动作值加入到下一时刻的环境变量中
            total_value_memory.append(total_value)
            if total_value <= 30:
                if act_number == 0:
                    q_max[0][0] = -20
                    q_max[0][1] = r1
                    total_value = 0
                else:
                    q_max[0][0] = r0
                    q_max[0][1] = -20
                    total_value = 0
            elif total_value >= 40:
                if act_number == 0:
                    q_max[0][0] = 20
                    q_max[0][1] = r1
                    total_value = 0
                else:
                    q_max[0][0] = r0
                    q_max[0][1] = 20
                    total_value = 0
            else:
                q_max[0][0] = r0 + l * model_replay.predict(fit_o)[0][
                    np.argmax(model_action.predict(fit_o))]  ##q值的目标值，或者可以称为label
                q_max[0][1] = r1 + l * model_replay.predict(fit_o)[0][np.argmax(model_action.predict(fit_o))]

            memory = np.vstack((memory, fit_i))  ##将此时刻处理过的环境值加入到环境记忆中
            q_memory = np.vstack((q_memory, fit_o))  ##将此时刻的q最大值加入到label记忆中
            action_memory.append(act_number)  ##将动作添加到动作记忆中
            total_value_memory.append(total_value)
            if total_value <= 30:
                if act_number == 0:
                    reward_buffer[0][0] = -20
                    reward_buffer[1][0] = r1
                else:
                    reward_buffer[0][0] = r0
                    reward_buffer[1][0] = -20
            elif total_value >= 40:
                if act_number == 0:
                    reward_buffer[0][0] = 20
                    reward_buffer[1][0] = r1
                else:
                    reward_buffer[0][0] = r0
                    reward_buffer[1][0] = 20
            else:
                reward_buffer[0][0] = r0
                reward_buffer[1][0] = r1
            r = np.reshape(reward_buffer, (1, 2, 1))
            reward_memory = np.vstack((reward_memory, r))  ##添加本次奖励到记忆中

            if j % 50 == 0:  ##每50次尝试之后对记忆中的数据进行训练
                q_replay = np.zeros((1, 2))  ## 记忆q值
                for times in range(0, 500):
                    k = choice(range(start_point, len(q_memory)))  ###将记忆中的数据再处理，使用新的权重得到新的目标q值
                    q = np.reshape(q_memory[k], (1, win, traini.shape[2] + 6, 1))
                    r2 = reward_memory[k][0]
                    r3 = reward_memory[k][1]
                    act_replay = action_memory[k]
                    if total_value_memory[k] <= 30 or total_value_memory[k] >= 40:
                        q_replay[0][act_replay] = reward_memory[k][act_replay]
                        q_replay[0][(act_replay + 1) % 2] = reward_memory[k][(act_replay + 1) % 2] + l * \
                                                            model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                    else:
                        q_replay[0][0] = r2 + l * model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                        q_replay[0][1] = r3 + l * model_replay.predict(q)[0][np.argmax(model_action.predict(q))]
                    m = memory[k]  ##这一时刻的输入
                    m = np.reshape(m, (1, win, traini.shape[2] + 6, 1))
                    model_replay.fit(m, q_replay, epochs=1, batch_size=1, verbose=2)  ###训练记忆模型
            if j % 500 == 0:
                weights = model_replay.get_weights()
                action_weights = model_action.get_weights()  ###将记忆模型的权重送给动作选择模型
                for i in range(0, len(action_weights)):
                    action_weights[i] = weights[i]
                model_action.set_weights(action_weights)
                start_point = start_point + 20

            model_replay.fit(fit_i, q_max, epochs=1, batch_size=1, verbose=2)  ##训练模型
            if j % 999 == 0:
                model_replay.save(model_name)

    def test_dqn(self,name,model_name):
        model_test = load_model(model_name)
        traini,testi,traino,testo = self.seperate_dqn_normal(name, 9, 10)
        win = 10
        a_test = np.zeros(2,)
        a_test[1] = 4
        total_value = 0
        price_test = data_processing.close_price(name)  ##价格
        r = len(price_test)//10
        price_test = price_test[r*9+2*win:]
        x_test = 0
        for i in range(1,len(testi)-win-1):
            d_test = i%2 ## 这一步在a中的位置
            c_test = (i+1)%2 ##上一步在a中的位置
            state_test = np.reshape(testi[i],(1,win,testi.shape[2],1))
            time = i%20 - 1
            data = self.add_data(win)       ##获取用于添加到测试集中的动作数据
            state_test = np.dstack((state_test,data[a_test[d_test]])) ##将上一次动作对应的数据添加到测试集输入中
            act_name, act = self.action(state_test,a_test[d_test],model_test,i,0.8)
            a_test[c_test] = act_name
            r0_test,r1_test,x_test = self.value(a_test,c_test,d_test,price_test,win,x_test,i)
            if act ==0:
                total_value = total_value + r0_test
            else:
                total_value = total_value + r1_test
        print("total value is",total_value)