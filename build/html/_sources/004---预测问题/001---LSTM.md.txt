# LSTM

## 概述

概览：

<img src="..\_static\2.png" alt="2" style="zoom:50%;" />

---

遗忘门：

<img src="..\_static\3.png" alt="2" style="zoom:35%;" />



输入门：

<img src="..\_static\4.png" alt="2" style="zoom:50%;" />



更新传输带：

<img src="..\_static\5.png" alt="2" style="zoom:60%;" />



输出门：

<img src="..\_static\6.png" alt="2" style="zoom:50%;" />



更新输出：

<img src="..\_static\7.png" alt="2" style="zoom:58%;" />

---

<img src="..\_static\9.png" alt="2" style="zoom:30%;" />

如上图：每个神经元的输入数据有3个，即输入特征数为3。X的不同下标代表不同时刻，即共有t+1个时间展开步。LSTM层一共有3层，其中，前两层的输出模式为`sequence`，最后一层的输出模式为`last`。

---

<img src="..\_static\8.png" alt="2" style="zoom:58%;" />

上图是训练数据的形状，对应的输入特征数(channel)为3，时间展开步(time)为300，输入样本数(batch)为5。以元胞数组的形式为网络传递数据。

---

## Matlab实现

- 单输入-单输出

```matlab
clear;clc;
%% 加载数据
x = sort(50*rand(1000,1));
data = besselj(0,x);
data = data';

%% 训练集归一化
mu = mean(data);
sig = std(data);
dataStandardized = (data - mu) / sig; 

lag = 60;
XTrain = {};
YTrain = [];
for ii=1:length(dataStandardized)-lag
    XTrain{ii} = dataStandardized(ii:ii+lag-1);
    YTrain(ii) = dataStandardized(ii+lag);
end

%% 建立神经网络层
layers = [ sequenceInputLayer(1,"Name","input")
           lstmLayer(100,"Name","lstm1","OutputMode","sequence")
           dropoutLayer(0.2,"Name","drop1")
           lstmLayer(50,"Name","lstm2","OutputMode","last")
           dropoutLayer(0.2,"Name","drop2")
           fullyConnectedLayer(1,"Name","fc")
           regressionLayer ];

%% 配置训练参数
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'MiniBatchSize',200,...
    'Plots','training-progress');

%% 训练网络
net = trainNetwork(XTrain,YTrain',layers,options); 

%% 验证
net = resetState(net);
[net,YTest] = predictAndUpdateState(net,XTrain,'MiniBatchSize',1);
YTest = sig*YTest + mu;

%% 预测
pre_step = 20;
net = resetState(net);
YPred = dataStandardized(lag+1:end)';
num_data = length(YPred);
for ii=1:pre_step
    tem_data = {YPred(end-lag+1:end)'};
    [~,YPred(ii+num_data)]=predictAndUpdateState(net,tem_data,'MiniBatchSize',1);
end
YPred = sig*YPred + mu;

%% 画图
hold on
plot(data(lag:end),'b')
plot(YTest,'g',LineWidth=2)
plot(num_data:num_data+pre_step-1,YPred(num_data+1:end),'r',LineWidth=2)
```

- 多输入-单输出

```matlab
clear;clc;
%% 加载数据
t = (0.01:0.01:20)';
a = (t.^2 - 10 * t);
b = cos(5*pi*t).*exp(-5*pi*0.001*t);
c = cos(7*pi*t).*exp(-7*pi*0.001*t);
y = exp(a*0.01).*c.*b;

%% 数据可视化
subplot(4,1,1);plot(t,a);ylabel('a');
subplot(4,1,2);plot(t,b);ylabel('b');
subplot(4,1,3);plot(t,c);ylabel('c');
subplot(4,1,4);plot(t,y);ylabel('y');

%% 划分训练集与测试集
data = [a,b,c,y];
numTimeStepsTrain = floor(0.9*numel(y));
dataTrain = data(1:numTimeStepsTrain+1,:);
dataTest = data(numTimeStepsTrain+1:end,:);

%% 训练集归一化
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - ones(size(dataTrain)).*mu) ./ (ones(size(dataTrain)).*sig); 
XTrain = dataTrainStandardized(:,1:3)';
YTrain = dataTrainStandardized(:,4)';

%% 建立神经网络层
layers = [sequenceInputLayer(3,"Name","input")
          lstmLayer(200,"Name","lstm")
          dropoutLayer(0.1,"Name","drop")
          fullyConnectedLayer(1,"Name","fc")
          regressionLayer("Name","regressionoutput")];

%% 配置训练参数
options = trainingOptions('adam', ...
                          'MaxEpochs',80, ...
                          'GradientThreshold',1, ...
                          'InitialLearnRate',0.005, ...
                          'LearnRateSchedule','piecewise', ...
                          'LearnRateDropPeriod',125, ...
                          'LearnRateDropFactor',0.2, ...
                          'Verbose',0, ...
                          'Plots','training-progress'); 

%% 训练网络
net = trainNetwork(XTrain,YTrain,layers,options); 

%% 测试集数据归一化
dataTestStandardized = (dataTest - ones(size(dataTest)).*mu) ./ (ones(size(dataTest)).*sig); 
XTest = dataTestStandardized(:,1:3)';

%% 多步预测
net = predictAndUpdateState(net,XTrain);
numTimeStepsTest = numel(XTest(1,:));
YPred = [];
for i = 1:numTimeStepsTest
    [net,YPred(i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end 

%% 预测值反归一化
YPred = sig(4)*YPred + mu(4);

%% 结果可视化
idx1 = 1:numel(y);
idx2 = (numTimeStepsTrain+1):(numTimeStepsTrain+numTimeStepsTest);

figure
subplot(2,1,1)
hold on
plot(idx1, data(1:end,4),'-k')
plot(idx2, YPred,'-r',LineWidth=1.2)
hold off
xlabel("t")
ylabel("y")

subplot(2,1,2)
hold on
plot(idx1, data(1:end,4),'-k')
plot(idx2, YPred,'-r',LineWidth=1.2)
hold off
xlim([1800,2000])
xlabel("t")
ylabel("y")
title("Forecast")
legend(["Observed" "Forecast"]) 
```

<img src="..\_static\10.png" alt="2" style="zoom:58%;" />

注：以上内容参考了[MATLAB深度学习之LSTM 参数理解](https://www.bilibili.com/video/BV1ZS4y1P7R6/?spm_id_from=333.788)等系列视频！

---

## Python实现

```{tip}
数据集：
{download}`DOM_hourly<../_static/DOM_hourly.csv>`
{download}`BikeShares<../_static/BikeShares.csv>`
```

- 单输入-单输出

```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

## 数据预处理
# 加载数据
dataset = pd.read_csv('DOM_hourly.csv',parse_dates=['Datetime'],index_col=['Datetime'])
dataset.head(3)
# 数据归一化
scaler = MinMaxScaler()
dataset['DOM_MW']=scaler.fit_transform(dataset['DOM_MW'].values.reshape(-1,1))
dataset.head(3)

## 特征工程
def create_new_dataset(dataset,seq_len=20):
    start = 0 
    end = dataset.shape[0] - seq_len
    X=[]
    y=[]
    for i in range(start,end):
        features = dataset[i:i+seq_len]
        labels = dataset[i+seq_len]
        X.append(features)
        y.append(labels)
    return np.array(X), np.array(y)
def split_dataset(X,y,train_radio=0.9):
    train_num = int(len(X) * train_radio)
    X_train = X[:train_num]
    y_train = y[:train_num]
    X_test = X[train_num:]
    y_test = y[train_num:]
    return X_train,y_train,X_test,y_test
def creat_batch_data(X,y,batch_size,mytype=1):
    if mytype==1:
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
        train_batch_dataset = dataset.cache().shuffle(1000).batch(batch_size)
        return train_batch_dataset
    if mytype==2:
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
        test_batch_dataset = dataset.batch(batch_size)
        return test_batch_dataset
# 创建特征数据集
seq_len = 20
X, y = create_new_dataset(dataset.values, seq_len=seq_len)
# 划分训练集和测试集
X_train,y_train,X_test,y_test = split_dataset(X,y,train_radio=0.9)
# 获取批数据
train_batch_dataset = creat_batch_data(X_train, y_train, batch_size=256, mytype=1)
test_batch_dataset = creat_batch_data(X_test, y_test,batch_size=256, mytype=2)
# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(8,input_shape=(seq_len,1)),
    tf.keras.layers.Dense(1)
])
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='best_checkpoint.hdf5',
                                                         monitor='loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         save_weights_only=True)
# 模型编译
model.compile(optimizer='adam',loss='mae')
# 模型训练
history = model.fit(train_batch_dataset,
                    epochs=5,
                    validation_data=test_batch_dataset,
                    callbacks=checkpoint_callback)
# 模型验证
test_pred = model.predict(X_test, verbose=1)
score = r2_score(y_test,test_pred)
print('r^2=%.2f'%score)
# 模型预测
def predict_next(model,sample,num=10):
    temp1 = list(sample[:,0])
    for i in range(num):
        sample = sample.reshape(1,seq_len,1)
        pred = model.predict(sample)
        value = pred.tolist()[0][0]
        temp1.append(value)
        sample = np.array(temp1[i+1:seq_len+i+1])
    return temp1
true_data = X_test[-1]
preds = predict_next(model,true_data,15)
# 绘图
plt.figure(figsize=(6,4))
plt.plot(preds, color='red')
plt.plot(true_data, color='black')
plt.show()
```

- 多输入-单输出

```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## 数据预处理
# 加载数据
dataset = pd.read_csv('BikeShares.csv',parse_dates=['timestamp'],index_col=['timestamp'])
dataset.head(3)
# 数据归一化
columns = ['cnt','t1','t2','hum','wind_speed']
for col in columns:
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1,1))

## 特征工程
def create_dataset(X,y,seq_len=10):
    start = 0 
    end = len(X) - seq_len
    features=[]
    targets=[]
    for i in range(start,end,1):
        data = X.iloc[i:i+seq_len]
        label = y.iloc[i+seq_len]
        features.append(data)
        targets.append(label)
    return np.array(features),np.array(targets)

def create_batch_dataset(X,y,train=True,buffer_size=1000,batch_size=128):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)
X = dataset.drop(columns = ['cnt'], axis=1)
y = dataset['cnt']
# 划分训练集和测试集
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False, random_state=666)
# 构造特征数据集
seq_len = 10
train_dataset,train_labels=create_dataset(X_train,y_train,seq_len=seq_len)
test_dataset,test_labels=create_dataset(X_test,y_test,seq_len=seq_len)
# 构造批数据
train_batch_dataset=create_batch_dataset(train_dataset,train_labels,batch_size=128)
test_batch_dataset=create_batch_dataset(test_dataset,test_labels,batch_size=128,train=False)

## 模型搭建
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=256, input_shape=(seq_len, 8),return_sequences=True),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(units=256, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(units=128, return_sequences=True),
    tf.keras.layers.LSTM(units=32),
    tf.keras.layers.Dense(1)
])
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='best_checkpoint.hdf5',
                                                         monitor='loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         save_weights_only=True)

## 模型编译
model.compile(optimizer='adam',loss='mse')

## 模型训练
history = model.fit(train_batch_dataset,
                    epochs=5,
                    validation_data=test_batch_dataset,
                    callbacks=[checkpoint_callback])

## 模型验证
test_preds = model.predict(test_dataset, verbose=1)
test_preds = test_preds[:, 0]
score = r2_score(test_labels,test_preds)
print('r^2=%.2f'%score)

## 绘图
plt.figure(figsize=(6,4))
plt.plot(test_labels[:300], label="True value")
plt.plot(test_preds[:300], label="Pred value")
plt.legend(loc='best')
plt.show()
```

注：以上内容参考了[TensorFlow 2.0 基于LSTM单变量预测_电力消耗案例](https://www.bilibili.com/video/BV1f5411K7qD?spm_id_from=333.1007.top_right_bar_window_default_collection.content.click)等系列视频！

  





