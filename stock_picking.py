import data_processing
import price_prediction
import reinforcement_learning2
import os
from keras.models import load_model
import numpy as np

os.chdir(r'C:\cointree')
string=['aapl','amzn','blk','fb','googl','gs','jpm','msft','orcl','tsla']

class Stock_picking:

    def BBIBOLL_fuction(self,data, number1, number2, number3, number4):
        p1 = data_processing.EMA_FUNCTION(data, number1)
        p2 =  data_processing.EMA_FUNCTION(data, number2)
        p3 = data_processing.EMA_FUNCTION(data, number3)
        p4 = data_processing.EMA_FUNCTION(data, number4)
        bbiboll = np.zeros((len(p4), 1))
        for i in range(0, len(p1)):
            s = p1[i][0] + p2[i][0] + p3[i][0] + p4[i][0]
            s = s / 4
            bbiboll[i][0] = s
        return bbiboll


    def rsi_e(self,data, number):
        close = data[:,3]
        diff = np.zeros((len(close), 1))
        up = np.zeros((len(close), 1))
        down = np.zeros((len(close), 1))
        ema_d = np.zeros((len(close), 1))
        ema_u = np.zeros((len(close), 1))
        rsi = np.zeros((len(close), 1))
        ##calculate the up and down value
        for i in range(0, len(close)):
            if i == 0:
                q = 0
                diff[i][0] = q
            else:
                q = close[i] - close[i - 1]
                diff[i][0] = q
        for i in range(0, len(diff)):
            if diff[i][0] > 0:
                s = diff[i][0]
                down[i][0] = 0
                up[i][0] = s
            else:
                t = (-1) * diff[i][0]
                up[i][0] = 0
                down[i][0] = t
        ##define the calculation part
        for i in range(0, len(up)):
            ## define the value for the first position
            if i == 0:
                a = 0
                ema_d[i][0] = a
                ema_u[i][0] = a
            ##define the value for the ema_u and ema_d
            else:
                ema_d[i][0] = ema_d[i - 1][0] * (number - 1) / (number + 1) + down[i][0] * (2) / (number + 1)
                ema_u[i][0] = ema_u[i - 1][0] * (number - 1) / (number + 1) + up[i][0] * (2) / (number + 1)
        ##define the value for the rsi
        for i in range(0, len(rsi)):
            if i == 0:
                rsi[i][0] = 0
            else:
                r = ema_u[i][0] / (ema_u[i][0] + ema_d[i][0])
                rsi[i][0] = r

        return rsi
    def rl_data(self,data):
        bbiboll = self.BBIBOLL_fuction(data,3,6,12,24)
        up,down = data_processing.UP_DOWN(bbiboll,10,4)
        rsi = self.rsi_e(data, 14)
        price = data[:,3]
        new_data = np.zeros((price.shape))
        for i in range(0, len(new_data)):
            if up[i] - down[i] != 0:
                new_data[i] = (price[i] - down[i]) / (up[i] - down[i])
                new_data[i] = (new_data[i])
                if new_data[i] > 1:
                    new_data[i] = 1.1
                elif new_data[i] < 0:
                    new_data[i] = -0.1
            else:
                new_data[i] = 0.5
        rl_data = np.hstack((new_data, rsi))
        return rl_data

    def stock(self,name,win,days):
        action = []
        price = []
        original_data = data_processing.load_data(name)
        original_data = np.delete(original_data,-1,axis = 1)
        buffer_data = original_data[:win]
        input = np.zeros((win, buffer_data.shape[1]))
        input_buffer = np.zeros(1,buffer_data.shape[1])
        for i in range(0, days):    #####预测20天的价格
            rl_original = buffer_data[i:]
            for j in range(0, win):
                for u in range(0, buffer_data.shape[1]):
                    t = buffer_data[i + j][u]
                    if buffer_data[i][u] != 0:
                        s = t / buffer_data[i][u]
                    else:
                        s = 1
                    s = s - 1
                    input[i][j][u] = s

            lstm_high = load_model('lstm_stock_high.h5')
            lstm_low = load_model('lstm_stock_low.h5')
            lstm_open = load_model('lstm_stock_open.h5')
            lstm_close = load_model('lstm_stock_close.h5')
            agent = load_model('dqn_stock_picking.h5')

            pred_high = lstm_high.predict(input)
            pred_low = lstm_low.predict(input)
            pred_open = lstm_open.predict(input)
            pred_close = lstm_open.predict(input)

            pred_open = (pred_open+1)*buffer_data[i][0]
            pred_high = (pred_high+1)*buffer_data[i][1]
            pred_low = (pred_low + 1) * buffer_data[i][2]
            pred_close = (pred_close + 1) * buffer_data[i][3]

            input_buffer[0][0] = pred_open
            input_buffer[0][1] = pred_high
            input_buffer[0][2] = pred_low
            input_buffer[0][3] = pred_close

            rldata = self.rl_data(rl_original)
            act = agent.predict(rldata)
            action.append(act)
            price.append(pred_close)
            buffer_data = np.vstack((buffer_data,input_buffer))

        return action,price



