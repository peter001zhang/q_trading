# 基于深度学习和强化学习的量化交易系统（quantitative trading system based on Deep learning and Reinforcement Learning）
(中文版本在英文版本的下面，请下拉查看)

## OUTLINE:
1. main structure

2. still working on 

3. conclusion


## MAIN STRUCTURE:
#### **The system consist of:**

**Data processing module**

**Price prediction module**

**The reinforcement learning module based on:**

the design for 6 actions(sell, short, sell_hold, short_hold, sell, cover)

**The reinforcement learning module based on:**

1. Using the up and down line of VWAP or BBIBOLL to transform the price into (-1,1)
2. Design two reinforcement learning models seperately for (buy, sell hold) and (short, cover,hold)
3. Set the priority for two models, and decide which action should be output

**Stocking picking strategy based on price prediction and RL return**
### Data Processing Module
I get the data set from Kaggle, which is the daily price and volume data of American stock market. The data set consist of open price, close price, high price, low price and volume. But this kind of data doesn't work well in the training of Deep learning and Reinforcement Learning. So I create dozens of technical analysis function to generate more feature for the input. Through this way, it could make the DQN agent easier to understand the input and increase the convergence speed.

Here I choose the most popular technical analysis method on Thinkorswim and in China. And make it easy to generate the data based on original data set.

The technical analysis method this module support are:

1. SMA 
2. EMA 
3. MACD_DEA
4. MACD_DIF
5. MACD_BAR
6. VWAP
7. VWAP_UP
8. VWAP_DOWN
9. BBIBOLL
10. BBIBOLL_UP
11. BBIBOLL_DOWN
12. RSI_EMA
13. RSI_SMA
14. TRIX
15. TMA
16. BIAS

![thinkorswimimage](https://github.com/peter001zhang/q_trading/blob/master/image/thinkorswim.png)

If you don't understand the calculation formula, please search in Google.

Besides, in this module, I also define a function whose inputs are data for two technical line(a,b), and the outputs are four numbers(-1,-0.5,0.5,1). "-1" means a came above b from the below, "-0.5" means a is alway bigger than b, "0.5" means b is always bigger than a, and "1" means b came above a from below. this could be used for the analysis of MACD.

Through the creation 17 technical analysis methods, I largely increase the number of features. Rightnow, I can only get the daily stock price data, so the data set is so small, and it is hard to see the advantage of this. But if I could use the 5min data or 1 min data, I believe the training result will be better because of the technical lines.

### Price Prediction Module



In this module, we first separate the data into training_input,training_output, testing_input and testing_output based on different inputs_shape of XGBOOST, LSTM and CNN. Then we do the normalization for each data set.

In this part, the function supported are: 
1. separate the data for ML methods, separate the data for LSTM, seperate the data for CNN
2. separate and normalized the data for ML methods, separate and normalized the data for LSTM, seperate and normalized the data for CNN

The normalization method is calculate the proportion between each data in the windows and the first data in the windows.

The price prediction models are:XGBOOST(REGRESSION), SVR, GRU, LSTM,modified Deepsense, and Deepsense with Attention Model.

According to the experiment, I discover that SVR did the worst job, and XJBOOST also couldn't do the job very well. But the training result of XJBOOST depends largly on the adjustment of parameters, and I am not very good at it, so it maybe better after the adjustment. GRU could be understanded as the simplify LSTM, it got higher speed, but the result is not as good as LSTM.

![deepsense_image](https://github.com/peter001zhang/q_trading/blob/master/image/deepsense.jpg)

Deepsense network is Deep learning structure used on mobile sensor's data analysis. In my point of view, the fluctuation of stock price looks very simillar with the sensor's data. So I adjusted the structure of Deepsense to make it more suitable for my data set. 

![attention_model_image](https://github.com/peter001zhang/q_trading/blob/master/image/am.jpg)

The reason for adding an Attention Model is that I have seen an paper about face recognition, telling that using 1*1 convolution kernels for three times, and multiply is with the original image, could increase the weights for the most important features after training, and those important features are nose, eyes, mouths. So I am trying to use the same ideas to get the most important feature information from the input_data.

The final experiment shows that the Deepsense model with attention model perform better than the original Deepsense model. Though the reult is also not as good as the LSTM, I can still expected to get better result after using the 5min data and 1min data.

![lstm_image](https://github.com/peter001zhang/q_trading/blob/master/image/lstm.jpg)

In a word, the LSTM gets the best results, and the avearge R2 score for 10 stocks could be higher than 90%.


### Reinforcement Learning Module 1 

In this module, I use simple DQN, DDQN and Dueling DDQN to realize my ideas.

My idea is considering the actions of long and short together, which means choose an action from (buy, sell, hold, short, cover). But the limitations are too much, like: agent may output "short" continuously, or output "buy" continously, which are all not reasonable.

Through the research, I found out that I could separate the "hold" action into "hold_in_buy" and "hold_in_short". The advantage of doing that is I could reshape the last action into one-hot shape, and concatenated it into the inputs(state). So under this condition, the output at anytime will be only two:

buy: holdb, sell

holdb: holdb, sell

short: holds, cover

holds:holds, cover

sell: buy, short

cover: buy, short

Here you can see, I didn't set an action for hold while there is no portfolios. The reason for doing that is I wanted the agent to learn buy at low, sell at high, short at high and cover at low, so it is useless to set an action under this condition.

Key points:

1. The input consist of two parts:

    1. the data for technical analysis and original data generated from the data processing modules.

    2. the one-hot formation for the last action.

2. The action selection is a little bit different from the traditional DQN, because the output action of my NN is decided by the last action, so unlike the tradional DQN, which is the output of each nuerons of the last dense is the action, my action is chozen by the complex selection function. By the way, the last action is stored by a (2,1) queue.

3. As for the reward, I have to use the last functional price recording, and use this recording and the present price to calculate the reward. I use the price_buffer to record the price for the last buy and short, and clear it to zeros when sell and cover action is chosen.

4. Just like using DQN to play a game, I have to set a finish condition. If it finished, the Q value will be the same as the reward. Here, I define that under the condition of processing one share, if the return is higher than 40, or the loss is higher than 30, we finish it.

because the first data stored in the memory is naive and stupied, I define every 500 epoches in replay, the memory should delete the first 50 data.

The Reinforcement Models:

In this module, three models are supported:Simple DQN, DDQN and Dueling DDQN. In theory the Dueling DDQN should get the best result. But afterall, trading is not playing a game, so it is hard to say whether the tricks is usful in trading. So I keep all three models for testing.

Here I would love to breifly introduce three models:

DQN is the Deep Q network based on the idea of Q-learning, it replace the Q-TABLE with the CNN network. So it solved the capacity problems of Q-learning.

In my view, the Key point of DQN should be:

1. the remember and replay function. In my view of point, the method of randomly choosing previous records(remember) and use it for training is more like a real NN, and it gets better results of course.
2. it design a target net, which is an NN with later updated weights. And the target net is used for calculating the Qmax of next state, and it solves the problem that the connaction between Qmax and present one is too high.

DDQN is based on the DQN, the difference is the DDQN use the eval net to calculate the Qmax and get the corresponding action, and use the target net to calculate the Q(next state, action) and the traditional Qmax.

![dueling_image](https://github.com/peter001zhang/q_trading/blob/master/image/dueling.png)

Dueling DDQN is based on the DDQN. the idea is separate the state and the action in the NN, which means at the very last of NN model, we separate the action that supposed to be output into actions and state. Then substract the value of action with the mean of every actions. And finally added the state and the actions as the outputs. The reason for doing this is that in some state, whatever the action you do, there isn't any influence on the next state. So in the Quantitative trading, I can understand it like this, when your initial capital is not big enough, whatever the action you made, it can hardly effect the stock price.

### Reinforcement Learning Module 2

In the previous Reinforcement module I discover that because of the price fluctuation in daily data is not as high as day-trading, it will last a long period of bear of bull market, so the DQN agent will tend to choose buy at the start of bull market,and hold for a very long time. This is actually the right decision, but I want the agent to be more sensitive to the price, which mean through short and buy to earn money in the fluctuation of stock price.

So I design a new model to fix this, and the key point of this model are two parts:

![price_image](https://github.com/peter001zhang/q_trading/blob/master/image/price.png)

1. we understand that if the price data is concussion in a fix range, the reinforcement learning agent could learning the strategies better. So I use the the up and down line of VWAP(4std), to reshape the price data in to (-1,1). But even though I set the standard deviation as 4 times, there are still some price get out of the range, so I set those data into 1.1 and -1.1, which means out of the range.

2. I use two reinforcement learning models to design the action policies for (buy, sell, hold) and (short, cover, hold) separately. And the final output action is chosen under a principle that the priority for buy_model is higher than the short_model. This principle means that we choose the action output by the buy_model first, then the short_model. This design may not be that reasonable, so I am trying to use the NLP and GOOGLE TREND data, to make a prediction of the future trend, and use the result for adjusting the priority.

Unlike the previous reinforcement learning module, the finish condition and reward policy has all been changed, as well as the action selection:

1. finish condition: here I define every (buy,sell) action means the finish mark, vice versa.
2. reward policy: I define the reward as the difference between price datas which has already been reshaped into (-1,1)
3. action selection: here the selection is not as complex as the previous one, but I define the unreasonable action will be outputed as "hold".Then the action will get a punishment value.

The chosen for models is the same as the previous module, including the DQN, DDQN and Dueling DDQN, although the code maybe different, but the fundamental ideas are the same.

### Stocking Picking Strategy Based on Price Prediction and RL Return

I use the LSTM to predict the price data in a period of future, and give the price data as input to the reinforcement learning model, calculate the return. The stock with the highest return will be the chosen one. 

Because the VWAP need the volume data to make the calculations, and the volume data is really hard to predict, so I use the BBIBOLL to reshape the price data.

The idea is really simple, so I won't tell it in details.



## STILL WORKING ON:
**DDPG model**: the code has been finished, and I am testing it right now and will put it into the module as soon as possible.

**NLP**: the present idea is using the web crawler to get the title and first paragraph of the article in financial news. Then using the stocking price variety as the label to train to NN.

![trendimage](https://github.com/peter001zhang/q_trading/blob/master/image/trend.png)

**GOOGLE TREND**: there is a paper talk about using the GOOGLE TREND to make the predictions, but after several simple test, I found out the result isn't good enough. But I will keep trying.

**Reinforcement learning priority**: with the prediction result of NLP model and GOOGLE TREND model, I can adjust the priority of the second reinforcement learning module.



## CONCLUSION:
I spent lots of time and energy on the system, but meanwhile I have learned lots of knowledge, not only the technical knowledge, but more important, is the financial and quantitative trading knowledge. 

I tried to use the Thinkorswim and investopedia's paper money system, and learned alot about stock, futures and Options. And I also asked questions to popular day trader, and learned alot about the tricks of technical analysis, and those tricks are very important in building the reinforcement learning models.

In a word, the system is not able to get stable profit, but it has made a great improvement on myself. If you are interest, you can have a look, thanks for reading!

# 中文版本

## 目录：
1.主体结构

2.进行中的工作

3.总结

## 主体结构：
**系统包括**：

数据处理模块

价格预测模块

将(sell, short, sell_hold, short_hold,sell,cover)6个动作整体设计的强化学习模块

将价格数据通过VWAP或者BBIBOLL的UP和DOWN线，转换成在(-1,1)区间的震荡数据，同时将(buy, sell, hold)和(short, cover, hold) 分成两个强化学习模型分别设计，再通过设立优先级，输出动作的强化学习模块，一定程度上解决了上一个模块对小幅价格波动不敏感的问题

依据价格预测和强化学习回报的选股策略模块

### 数据处理模块：
我使用kaggle上获得的美国股市每日股票价格和交易量数据，包括开盘价，收盘价，最高价，最低价和交易量。但是这样的数据并不能很好的用于深度学习和强化学习模型的训练，因此我通过为原数据创建相对应的技术指示线丰富输入的特征数量，同时通过这种方法也可以使得强化学习对输入的理解更加简单，加速收敛。

这里我选择在thinkorswim中和国内常用的技术指示线，并使之能够根据原数据的值自动生成对应的数据。

支持创建指示线数据的方法有：
1. SMA 
2. EMA 
3. MACD_DEA
4. MACD_DIF
5. MACD_BAR
6. VWAP
7. VWAP_UP
8. VWAP_DOWN
9. BBIBOLL
10. BBIBOLL_UP
11. BBIBOLL_DOWN
12. RSI_EMA
13. RSI_SMA
14. TRIX
15. TMA
16. BIAS

![thinkorswimimage](https://github.com/peter001zhang/q_trading/blob/master/image/thinkorswim.png)

指示线数据的计算方法请自行百度，这里不再赘述。

除此之外，在这个模块中我还定义了一个函数，它的输入是两条指示线（a,b）的数据，输出则是4个值（-1，-0.5,0.5,1），分别对应a在上一点小于b，但是这一点的值大于b；a在上一点的值和这一点的值都大于b；b在上一点和这一点的值都大于a；b在上一点的值小于a，但是在这一点的值大于a。这样我们就可以获得的向上穿过点和向下穿过点的数据。

通过以上17种指示线数据的创建，可以极大的提升特征数量，现在我能拿到的是天级的数据，数据量过小，效果不是十分明显。但若是使用5min或者1min的数据，相信会对模型的训练效果有所提升。


### 价格预测模块：
在这个模块中，我们首先根据LSTM，CNN，XGBOOST的输入格式对经过数据处理模块整理过的数据集进行归一化和分割。

支持的模块有：为ML分割，为ML分割并进行归一化，为LSTM分割，为lstm分割并进行归一化，为CNN分割，为CNN分割并进行归一化。

这里归一化的方法是通过计算windows中每一个价格值和windows中第一个价格数据的比例得到的。

这里的三种分割方式，我都通过一个单独的函数支持直接利用原数据进行分割和归一化，处理过的数据不包含指示线数据，可以方便进行模型测试。



价格预测的模块有：XJBOOST(REGRESSION),SVR,GRU,LSTM,修改了模型的Deepsense网络和添加了注意力模型的Deepsense模型。

根据实验，我发现SVR的效果是最差的，XJBOOST的效果也不是很好，但是XJBOOST的效果很吃调参，我对XJBOOST的研究并不够深刻，所以或许有提升的可能。GRU可以理解成一个简化的LSTM，在这里的测试阶段我发现虽然速度很快，但是效果比LSTM还是有点差距。

![deepsense_image](https://github.com/peter001zhang/q_trading/blob/master/image/deepsense.jpg)

Deepsense网络是一种应用于处理移动端传感器时序数据的深度学习框架，在我看来，股价的波动和传感器时序数据的波动有异曲同工的感觉，所以我根据数据集的特点，对原本框架进行修改并使用在价格预测系统中。为Deepsense模型添加注意力模型的原因是，我在一篇关于人脸识别的论文中看到的一个方法。为了加强鼻子，眼睛，嘴这些最重要的分类特征而使用1*1的卷积核对一张图片进行3次相同的卷积处理，之后与源图片逐位相乘，经过学习最终达到加大重要特征权重的目的。我利用这个思想对数据进行相同的处理，希望的到类似的效果。

![attention_model_image](https://github.com/peter001zhang/q_trading/blob/master/image/am.jpg)

最终效果显示，添加了注意力模型的Deepsense模型相比较原模型而言，效果更好，但是在数据量较小，训练次数较少的情况下依旧没有LSTM表现好。但是我可以期待在获取5min或者1min级别的数据之后，这个模型的效果或许会超过LSTM。

![lstm_image](https://github.com/peter001zhang/q_trading/blob/master/image/lstm.jpg)

总而言之，目前而言lstm的预测效果最好，平均R2能够达到0.9以上。

### 强化学习模块1：

在这个模块中，我使用simple DQN，DDQN和Dueling DDQN 的模型对自己的思路进行实现。

我的思路是将做多和做空的动作整体考虑，但是这样对动作的限制条件过多，比如agent可能会输出连续的short，或者连续的buy，或者在空仓连续的卖，这样生成的动作是十分不合理的。

通过研究发现，我将hold动作分成了holdb(buy时的hold)和holds(short时的hold)，这样做的好处是，我可以将上一次的动作整理成one-hot格式作为一个输入添加到这一时刻的环境中，这样可以选择的输出在任何时候都只有两个。

buy: holdb, sell

holdb: holdb, sell

short: holds, cover

holds:holds, cover

sell: buy, short

cover: buy, short

在这里可以发现我没有为空仓持有设立动作，原因是我希望agent能有学习到低点买入，高点卖出，高点卖空，低点平仓的策略，所以自然也就不需要为空仓时设立持有动作了。

重点内容：

1. 环境的输入包括两部分，第一部分是在数据处理模块中生成的指示线数据和原数据，经过分割和正规化处理之后作为环境的一部分输入，第二部分是上一个动作的one-hot格式，这在一定形式上告诉agent该选择什么样的动作。

2. 动作的选择和DQN的选择有所不同，因为神经网络输出的动作是根据前一个动作决定的，而不是像一般DQN，最后全连接层神经元的输出就是对应动作的Q值。所以我在动作选择的函数中通过条件选择输出正确的动作。在这里上一个动作通过一个2位的队列实现。

3. Reward的值需要引入价格记录，并通过价格记录与现在的价格做对比，决定为当前动作赋予的reward值，我们通过价格buffer记录上一次buy或者short时的价格记录，在sell或者cover时清零。

4. 像强化学习玩游戏一样，我需要设立一个停止条件，若停止，则Q值直接等于reward，不需要再加上下一次中的Qmax。这里，我设立的条件是在一次只买一股的情况下，获利40以上，或者损失30以上，则停止。

因为最早的数据是幼稚且愚蠢的，所以我们在replay中定义每500次迭代，就将最靠前的memory中的数据丢掉，不进行replay。

模型的选择：

这里支持3种模型，simple DQN,DDQN 和Dueling DDQN，理论上来说，应该是Dueling DDQN的效果最好，但是交易毕竟不是玩游戏，很难说用在游戏训练上的小技巧有没有用，所以我将三种模型都保留，为以后的测试做准备。

这里简单介绍一下三种模型：

DQN就是建立在Qlearning的基础之上开发的Deep Q network，通过使用神经网络替代Q值表，更加适用于不同的任务。

DQN的重点在我看来有两点：一是remember和replay模块，这将之前的数据随机选择拿来学习的概念在我看来更加符合神经网络训练的思路，效果也很好。二是引入了目标网络的概念，通过一个参数更新较慢并不直接参加训练的网络来对Qmax的值进行计算，很好的解决了目标计算与当前值相关性过高的问题。

DDQN就是在DQN的基础之上，进一步通过主体网络计算QMAX对应的动作，再由目标网络得到对应Q值的思想。

![dueling_image](https://github.com/peter001zhang/q_trading/blob/master/image/dueling.png)

Dueling DQN的思想则是将环境和动作分开考虑，具体表现在神经网络最后的输出，将本该直接输出的动作值分开成环境和动作，再将每个动作的值减去平均值，加上环境对应神经元的值，作为输出。这样做的原因是在有些环境下无论做什么样的动作，对下一个state都没有什么影响，在量化交易中，我理解成，在本金较低时，无论做什么样的交易动作，其实对股价的影响都是可以忽略不计的。

### 强化学习模块2：
在上一个思路中，我发现因为价格在每日的波动幅度不像日交易那么大，会维持一个较长的上升或者下降趋势(即牛市或者熊市)，所以DQNagent选择动作时倾向于在上升的初期选择买入，然后长时间持有，这样的做法当然无可厚非，但是我希望agent能够对价格参数更加的敏感，即在股价波动时也能通过short，buy的选择切换进行套利。

所以我设计了一个全新的模型来解决这个问题，这个模型最重要的思路有两个：

![price_image](https://github.com/peter001zhang/q_trading/blob/master/image/price.png)

1. 如果价格数据在一个固定的范围震荡，强化学习就能更好的学习买卖策略，我通过VWAP的UP线(这里我们设定为4倍标准差)和DOWN线给价格数据转换成在(-1,1)的震荡数据。但是即时我将标准差设为4倍，依旧会有价格超过1和-1的范围，我将这样的数据的值设为1.1和-1.1，理解为超过最高值。
2. 通过两个强化学习模型分别设计(buy,sell,hold)和(short,cover,hold)的动作，最后的动作输出依据做多模型的优先级比做空模型的优先级更高来进行选择（即先选择做多模型输出的动作，若动作不满足条件，后选择做空模型输出的动作），这样的设计不一定合理，比如在熊市，做空模型的优先级应该高于做多模型，所以我正在通过NLP技术和GOOGLE TREND的数据，对价格的大趋势进行预测，希望能够达到切换优先级的目的。

与上一个模型不同的是，这个模型的停止条件和奖励机制都有所改变，同时动作选择上也进行了大幅度简化：
1. 停止条件：这里我设定，停止条件为(buy,sell)或者（sell,buy）的区间为一次回合，每一次buy，sell出现都意味着回合的停止，对于做空模型一样。
2. 奖励机制：奖励我设定为映射到（-1,1）上的价格之间的差距。
3. 动作选择：动作选择上不像上一个模型那样复杂，而是规定了不合理的动作为hold，但是会接受一个惩罚值。


模型的选择一样，都是基于DQN的变种模型，包括simple DQN, DDQN和Dueling DDQN，代码实现上有所差距，但是思想不变，这里不做过多赘述。

### 依据预测价格和强化学习回报的选股策略模块：
这个模块， 故名思议就是通过LSTM预测未来一定时间的价格数据，同时将数据喂给强化学习模型，计算回报，回报最高股票就是通过我们的模型选择出来的股票。这里因为VWAP中涉及到交易量的数据，而交易量数据很难进行预测，所以我们通过BBIBOLL的UP和DOWN线进行价格的映射。

这个模块的思路较为简单就不做过多赘述了。

## 进行中的工作：
DDPG强化学习模型：代码已经实现，现在正处在测试阶段，相信不久之后就能添加到强化学习的模型中。

NLP：目前的思路是通过爬虫技术，获取financial news上文章的标题，副标题和第一段作为输入，通过之后一段时间的股价变动（上升，稳定和下降）作为标签进行训练，期望获得对较长时间范围内的趋势进行预测。

![trendimage](https://github.com/peter001zhang/q_trading/blob/master/image/trend.png)

GOOGLE TREND：有篇文章详细介绍了GOOGLE TREND在对股价趋势预测上的可行性，我已经进行过一些简单的测试，但是测试效果并不理想，但是依旧可以期望在对模型和输入的变动过程中，获得更好的结果。

强化学习模型优先级：有了GOOGLE TREND和NLP的预测数据我就可以根据这样的数据对强化学习第二种思路中做多，做空模型的优先级进行调整。

## 总结：
这个系统花费了我大量的时间和精力，但是同时，在搭建的过程中学习了很多知识，不仅仅是技术上的，更加重要的是学习了很多金融学，量化交易的知识。通过对investopedia和thinkorswim中paper money交易系统的尝试，对股票，期货，期权交易都有了更加深刻的认识，通过向youtube知名的日交易员请教，学习了很多技术指示线的使用套路，并将其思想融合进搭建的强化学习模型中，极大的开阔了我的思路，使我受益匪浅。

总而言之，这个系统还不是一个可以稳定盈利的系统，但是为我今后的设计工作打下了基础，各位有兴趣的也可以参考一下，谢谢阅读！








