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
dataTrain = data(1:numTimeStepsTrain+1,:);%【？】
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
net = predictAndUpdateState(net,XTrain);%【？】
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