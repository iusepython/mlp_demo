% this is a Matlab demo of Multilayer Perceptron
% author: hankai
% contact: kaihana@163.com
clear;clc;close all;

%% add code path
addpath('./code/assist_code');
addpath('./code/activation_function');
addpath('./code/pretrain_code');
addpath('./code/finetune_code');

%% load data
trainX = loadMNISTImages('./data/MNIST/train-images.idx3-ubyte');
trainLabel = loadMNISTLabels('./data/MNIST/train-labels.idx1-ubyte');
trainY = zeros(size(trainLabel,1),10);
for i = 1:size(trainLabel,1)
    label = trainLabel(i);
    trainY(i,label+1)=1;
end
testX = loadMNISTImages('./data/MNIST/t10k-images.idx3-ubyte');
testLabel = loadMNISTLabels('./data/MNIST/t10k-labels.idx1-ubyte');
testY = zeros(size(testLabel,1),10);
for i = 1:size(testLabel,1)
    label = testLabel(i);
    testY(i,label+1)=1;
end

%% train
% number of nodes of each layer
d1 = size(trainX,2);
d2 = 128;
d3 = 64;
d4 = size(trainY,2);
ds = {d1,d2,d3,d4};
% pretrain with sparse autoencoder
lambda = 3e-3;     % weight decay parameter       
beta = 3;          % weight of sparsity penalty term  
rho = 0.1;         % sparsity parameter
pretrain_max_iter = 10;
activation_type = 'sigmoid';
[Ws,bs] = pretrain(trainX,ds,lambda,beta,rho,pretrain_max_iter,activation_type);
display_network(Ws{1},12);  % display the weight map
% finetune using labelled data
lambda = 1e-4;  % weight decay parameter       
finetune_max_iter = 100;
activation_type = 'sigmoid';
[Ws,bs] = finetune(trainX,trainY,Ws,bs,ds,lambda,finetune_max_iter,'softmax',activation_type);
display_network(Ws{1},12); % display the weight map

%% save model
save ./model/mnist_mlp.mat Ws bs activation_type;

%% load model and test
load('./model/mnist_mlp.mat');
test_acc = accuracy(Ws,bs,testX,testY,activation_type)

